from dataclasses import asdict
import hashlib
import time
import numpy as np
import torch
from sionna.phy import config as sn_config
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid,ResourceGridMapper,LSChannelEstimator,ZFEqualizer,LMMSEEqualizer
from sionna.phy.channel.tr38901 import AntennaArray,UMi,UMa,RMa
from sionna.phy.channel import gen_single_sector_topology,subcarrier_frequencies,cir_to_ofdm_channel
from sionna.phy.mapping import Mapper,Demapper
from sionna.phy.fec.ldpc import LDPC5GEncoder,LDPC5GDecoder
from sionna.phy.utils import ebnodb2no
from .security import GGHCodec,hard_to_soft
from .receivers import mmse_sic

def synchronize(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)
    elif str(device).startswith('mps'): torch.mps.synchronize()

def timed(fn,device):
    synchronize(device); start=time.perf_counter(); result=fn(); synchronize(device)
    return result,(time.perf_counter()-start)*1000

def generator(seed,device): return torch.Generator(device=device).manual_seed(seed)
def complex_noise(shape,g,device):
    return torch.complex(torch.randn(shape,generator=g,device=device,dtype=torch.float64),torch.randn(shape,generator=g,device=device,dtype=torch.float64))/2**.5

def quantize(y,bits,load=4.):
    """Uniform mid-rise ADC per antenna/frame and per I/Q, clipping at `load` times the per-component RMS."""
    rms=(y.abs().square().mean((-2,-1),keepdim=True)/2).sqrt()
    step=2*load*rms/2**bits
    q=lambda v:((torch.floor(v/step)+.5)*step).clamp(-load*rms+step/2,load*rms-step/2)
    return torch.complex(q(y.real),q(y.imag))

class ExperimentModel:
    def __init__(self,c,key=None):
        self.c=c; kw={'precision':'double','device':c.device}
        sn_config.precision="double"
        sn_config.seed=c.seed
        self.rg=ResourceGrid(c.num_symbols,c.fft_size,c.subcarrier_spacing,num_tx=c.num_users,num_streams_per_tx=1,cyclic_prefix_length=c.cyclic_prefix_length,pilot_pattern='kronecker',pilot_ofdm_symbol_indices=list(c.pilot_indices),**kw)
        self.sm=StreamManagement(np.ones((1,c.num_users),dtype=int),1)
        self.grid_mapper=ResourceGridMapper(self.rg,**kw)
        self.mapper=Mapper('qam',c.bits_per_symbol,**kw)
        self.demapper=Demapper('app','qam',c.bits_per_symbol,**kw)
        self.encoder=LDPC5GEncoder(c.k_bits,c.n_bits,**kw) if c.coded else None
        self.decoder=LDPC5GDecoder(self.encoder,num_iter=c.ldpc_iterations,hard_out=True,**kw) if c.coded else None
        self.codec=GGHCodec(c,key) if c.security_mode!='none' else None
        if self.codec is None and key is not None: raise ValueError('Plain PHY must not carry a key')
        self.equalizer=(ZFEqualizer if c.receiver=='zf' else LMMSEEqualizer)(self.rg,self.sm,**kw)
        self.estimator=LSChannelEstimator(self.rg,interpolation_type='lin' if c.csi_mode=='ls_lin' else 'nn',**kw)
        ut=AntennaArray(num_rows=1,num_cols=1,polarization='single',polarization_type='V',antenna_pattern='omni',carrier_frequency=c.carrier_frequency,**kw)
        bs=AntennaArray(num_rows=1,num_cols=c.num_bs_ant//2,polarization='dual',polarization_type='cross',antenna_pattern='38.901',carrier_frequency=c.carrier_frequency,**kw)
        channel_kw=dict(carrier_frequency=c.carrier_frequency,ut_array=ut,bs_array=bs,direction='uplink',enable_pathloss=c.enable_pathloss,enable_shadow_fading=c.enable_shadow_fading,**kw)
        if c.scenario!='rma': channel_kw['o2i_model']='low'
        self.channel={'umi':UMi,'uma':UMa,'rma':RMa}[c.scenario](**channel_kw)
        self.data_indices=[i for i in range(c.num_symbols) if i not in c.pilot_indices]
        self.frequencies=subcarrier_frequencies(c.fft_size,c.subcarrier_spacing,**kw)

    def make_fixture(self,packet):
        """Independent random streams prevent encryption draws changing the channel."""
        c=self.c; seed=c.seed+100003*packet
        sn_config.seed=seed
        topology=gen_single_sector_topology(c.batch_size,c.num_users,c.scenario,min_ut_velocity=0,max_ut_velocity=0,precision='double',device=c.device)
        self.channel.set_topology(*topology)
        a,tau=self.channel(num_time_samples=c.num_symbols,sampling_frequency=1/self.rg.ofdm_symbol_duration)
        h=cir_to_ofdm_channel(self.frequencies,a,tau,normalize=not c.enable_pathloss)
        # Fixed-length mother bit buffer; coded comparisons use the same prefix.
        mother=torch.randint(0,2,(c.batch_size,c.num_users,1,c.n_bits),generator=generator(seed+1,c.device),device=c.device).to(torch.float64)
        noise=complex_noise((c.batch_size,1,c.num_bs_ant,c.num_symbols,c.fft_size),generator(seed+2,c.device),c.device)
        perturb=complex_noise(h.shape,generator(seed+3,c.device),c.device)
        evm=complex_noise((c.batch_size,c.num_users,1,c.num_symbols,c.fft_size),generator(seed+5,c.device),c.device)
        digest=hashlib.sha256(h.cpu().numpy().tobytes()+mother.cpu().numpy().tobytes()).hexdigest()[:20]
        return {'bits':mother,'h':h,'noise':noise,'csi_noise':perturb,'evm_noise':evm,'seed':seed,'fixture_id':digest}

    @torch.no_grad()
    def run_packet(self,ebno_db,packet=0,fixture=None):
        c=self.c; f=fixture if fixture is not None else self.make_fixture(packet)
        b=f['bits'][...,:c.k_bits]; no=ebnodb2no(ebno_db,c.bits_per_symbol,c.effective_rate,self.rg,precision="double",device=c.device)
        times={}; state=None; plain_scale=None
        def transmit():
            nonlocal state,plain_scale
            bits=self.encoder(b) if self.encoder else b
            x=self.mapper(bits)
            if self.codec:
                (x,state),times['encrypt_ms']=timed(lambda:self.codec.encrypt(x,generator(f['seed']+4,c.device)),c.device)
            else:
                times['encrypt_ms']=0.
                plain_scale=x.abs().square().mean(-1,keepdim=True).sqrt() if c.true_power_normalization else torch.ones_like(x.real[...,:1])
                x=x/plain_scale
            return self.grid_mapper(x),x
        (grid,x),times['tx_ms']=timed(transmit,c.device)
        if c.tx_evm_db is not None:
            # Transmitter distortion: Gaussian error at the given EVM on occupied REs, per user/frame.
            used=grid.abs()>0
            p=grid.abs().square().sum((-2,-1),keepdim=True)/used.sum((-2,-1),keepdim=True)
            grid=grid+used*(10**(c.tx_evm_db/10)*p).sqrt()*f['evm_noise']
        # Explicit common AWGN and common H across schemes; no RNG inside channel.
        h=f['h']; y=(h*grid[:,None,None,:,:,:,:]).sum((3,4))+no.sqrt()*f['noise']
        if c.adc_bits is not None: y=quantize(y,c.adc_bits)
        def receive():
            if c.csi_mode=='perfect': hh,err=h,torch.zeros((),device=c.device,dtype=torch.float64)
            elif c.csi_mode=='controlled_nmse':
                target=10**(c.csi_nmse_db/10)*h.abs().square().mean(dim=(-5,-4,-3,-2,-1),keepdim=True)
                e=f['csi_noise']; e=e/e.abs().square().mean(dim=(-5,-4,-3,-2,-1),keepdim=True).sqrt()
                hh,err=h+e*target.sqrt(),target
            else: hh,err=self.estimator(y,no)
            def detect():
                if c.receiver!='mmse_sic': return self.equalizer(y,hh,err,no)
                hm=hh[:,0,:, :,0,:,:].permute(0,3,4,1,2)
                ym=y[:,0].permute(0,2,3,1)
                ev=torch.broadcast_to(err,hh.shape)[:,0,:,:,0].sum(2).permute(0,2,3,1)
                points=None if self.codec else self.mapper.constellation.points[None,None,None,None,:]/plain_scale[:, :,0,0][:,None,None,:,None]
                out,nv=mmse_sic(ym,hm,no+ev,points)
                return tuple(t[:,self.data_indices].permute(0,3,1,2).reshape(c.batch_size,c.num_users,1,-1) for t in (out,nv))
            (xh,ne),times['receiver_ms']=timed(detect,c.device)
            errors={'wrong_babai_coordinates':0,'wrong_message_symbols':0}
            if self.codec:
                (xh,errors),times['decrypt_ms']=timed(lambda:self.codec.recover(xh,state),c.device)
                llr=hard_to_soft(self.demapper,xh,c.hard_llr_crossover)
            else:
                times['decrypt_ms']=0.; llr=self.demapper(xh*plain_scale,(ne*plain_scale.square()).clamp_min(1e-12))
            if self.decoder: bh,times['ldpc_ms']=timed(lambda:self.decoder(llr),c.device)
            else: bh=(llr>0).to(b.dtype); times['ldpc_ms']=0.
            return bh,ne,errors,float((hh-h).abs().square().mean()/h.abs().square().mean())
        (bh,ne,errors,csi_nmse),times['rx_ms']=timed(receive,c.device)
        if self.codec: errors=self.codec.diagnostics(errors,state)
        if bh.shape != b.shape: raise RuntimeError(f'Receiver shape {bh.shape} != information shape {b.shape}')
        wrong=b!=bh; blocks=wrong.any(-1); bit_errors=int(wrong.sum()); block_errors=int(blocks.sum())
        bler=float(blocks.double().mean()); duration=float(self.rg.ofdm_symbol_duration)*c.num_symbols
        delivered=(blocks.numel()-block_errors)*c.k_bits
        side_bits=64*c.num_users if c.true_power_normalization else 0
        # Scaling metadata is ideal, reliable side information. Charge its airtime
        # at the same nominal aggregate data rate, including pilots and CP.
        nominal=c.num_users*c.k_bits/duration; side_time=side_bits/nominal
        goodput=delivered/c.batch_size/(duration+side_time)
        row={**asdict(c),'scheme':c.scheme,'K':c.num_users,'M':c.num_bs_ant,'modulation':f'{2**c.bits_per_symbol}-QAM','csi':c.csi_mode,'sic_cancellation':('continuous' if self.codec else 'hard_qam') if c.receiver=='mmse_sic' else None,'tile':f'{c.tile_width}x{c.tile_time}' if self.codec else 'none','ebno_db':float(ebno_db),'packet':packet,'fixture_id':f['fixture_id'],
             'bit_errors':bit_errors,'num_bits':b.numel(),'block_errors':block_errors,'num_blocks':blocks.numel(),'ber':bit_errors/b.numel(),'bler':bler,'retransmission_probability':bler,'goodput':goodput,'effective_spectral_efficiency':goodput/(c.fft_size*c.subcarrier_spacing),'post_equalization_sinr':float((1/ne.clamp_min(1e-12)).mean()),'measured_csi_nmse':csi_nmse,'tx_data_power':float(x.abs().square().mean()),'tx_grid_power':float(grid.abs().square().mean()),'security_overhead_bits_per_frame':0,'normalization_sideinfo_bits_per_frame':side_bits,'ldpc_iterations_executed':c.ldpc_iterations if c.coded else 0,**errors,**times}
        row['normalization_metadata_ms']=side_time*1000
        row['total_ms']=times['tx_ms']+times['rx_ms']
        row['encrypt_ms_per_block']=times['encrypt_ms']/(c.batch_size*c.num_users*(c.fft_size*c.num_data_intervals/c.lattice_n)) if self.codec else 0.
        row['decrypt_ms_per_block']=times['decrypt_ms']/(c.batch_size*c.num_users*(c.fft_size*c.num_data_intervals/c.lattice_n)) if self.codec else 0.
        if self.codec:
            gamma=state['cipher_power']/state['message_power']
            row.update(self.codec.key.metadata)
            row['runtime_key_matrix_bytes']=sum(t.numel()*t.element_size() for t in (self.codec.B,self.codec.Ri,self.codec.Ui,self.codec.U))
            row.update(cipher_power=float(state['cipher_power'].mean()),gamma=float(gamma.mean()),gamma_db=float(10*torch.log10(gamma.mean())),cipher_power_db=float(10*torch.log10(state['cipher_power'].mean())))
        else:
            row.update(key_id=None,keygen_ms=0.,hr_R=None,hr_B=None,rho_R=None,sigma=None,public_key_bytes=0,private_key_bytes=0,intermediate_matrix_bytes=0,cipher_power=None,cipher_power_db=None,gamma=None,gamma_db=None)
        return row
