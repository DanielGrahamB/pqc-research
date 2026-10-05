"""Small deterministic validation gate; no research claims inferred from tests."""
from dataclasses import replace
import json
from pathlib import Path
import torch
from .config import ExperimentConfig
from .model import ExperimentModel,generator,complex_noise
from .security import KeyRegistry,GGHCodec
from .receivers import mmse_sic

def identity_fixture(model,packet=0,noiseless=True):
    c=model.c; seed=c.seed+100003*packet
    h=torch.eye(c.num_bs_ant,c.num_users,dtype=torch.complex128,device=c.device)[None,None,:,:,None,None,None].expand(c.batch_size,1,c.num_bs_ant,c.num_users,1,c.num_symbols,c.fft_size).clone()
    shape=(c.batch_size,1,c.num_bs_ant,c.num_symbols,c.fft_size)
    noise=complex_noise(shape,generator(seed+2,c.device),c.device)
    return {'h':h,'bits':torch.randint(0,2,(c.batch_size,c.num_users,1,c.n_bits),generator=generator(seed+1,c.device),device=c.device).double(),'noise':noise*0 if noiseless else noise,'csi_noise':torch.zeros_like(h),'seed':seed,'fixture_id':f'identity-{seed}'}

def validate(registry=None,output=None,device='cpu'):
    from .runner import fingerprint
    registry=registry or KeyRegistry(); checks=[]
    def check(name,condition,**evidence):
        checks.append({'test':name,'passed':bool(condition),**evidence})
        print(name, 'PASS' if condition else 'FAIL',flush=True)
    ref=ExperimentConfig(batch_size=2,device=device)
    key=registry.get(512,ref.seed)
    for q in (2,4,6):
        for mode,w,t in [('ggh_full',128,4),('ggh_subband',64,8)]:
            c=replace(ref,bits_per_symbol=q,security_mode=mode,tile_width=w,tile_time=t)
            model=ExperimentModel(c,key); f=identity_fixture(model)
            x=model.mapper(f['bits']); codec=model.codec
            tx,state=codec.encrypt(x,generator(123,device)); rx,err=codec.decrypt(tx,state)
            check(f'exact_lattice_{q}_{mode}',torch.allclose(x,rx,atol=1e-8,rtol=0) and not any(err.values()),**err)
            check(f'unit_power_{q}_{mode}',abs(float(tx.abs().square().mean())-1)<1e-10)
    for coded in (False,True):
        for mode,w,t in [('none',128,4),('ggh_full',128,4),('ggh_subband',64,8)]:
            for receiver in ('zf','lmmse','mmse_sic'):
                c=replace(ref,coded=coded,coderate=.5 if coded else 1.,security_mode=mode,tile_width=w,tile_time=t,receiver=receiver)
                m=ExperimentModel(c,key if mode!='none' else None)
                r=m.run_packet(120,fixture=identity_fixture(m))
                check(f'noiseless_{c.scheme}_{receiver}',r['bit_errors']==0,ber=r['ber'],num_bits=r['num_bits'])
    # Hard-to-soft must preserve noiseless labels across every supported QAM.
    for q in (2,4,6):
        from .security import hard_to_soft
        m=ExperimentModel(replace(ref,bits_per_symbol=q));f=identity_fixture(m)
        llr=hard_to_soft(m.demapper,m.mapper(f['bits']),ref.hard_llr_crossover)
        check(f'hard_to_soft_{q}',torch.equal(llr>0,f['bits'].bool()))
    # AWGN reference tests use equal Eb/N0, common mother bits and common noise.
    coding_evidence=[]
    for coded in (False,True):
        c=replace(ref,batch_size=8,coded=coded,coderate=.5 if coded else 1.)
        m=ExperimentModel(c); rows=[m.run_packet(5,packet=i,fixture=identity_fixture(m,i,False)) for i in range(2)]
        coding_evidence.append({'coded':coded,'ber':sum(r['bit_errors'] for r in rows)/sum(r['num_bits'] for r in rows),'bler':sum(r['block_errors'] for r in rows)/sum(r['num_blocks'] for r in rows)})
    check('conventional_ldpc_gain',coding_evidence[1]['ber']<coding_evidence[0]['ber'] and coding_evidence[1]['bler']<coding_evidence[0]['bler'],observations=coding_evidence)
    for receiver in ('zf','lmmse'):
        c=replace(ref,receiver=receiver);m=ExperimentModel(c)
        low=m.run_packet(-5,fixture=identity_fixture(m,noiseless=False));high=m.run_packet(20,fixture=identity_fixture(m,noiseless=False))
        check(f'plain_{receiver}_snr_trend',high['ber']<low['ber'] and high['ber']<.01,low=low['ber'],high=high['ber'])
    # Real TR38.901, all scenarios and CSI paths; also verifies paired randomness.
    for scenario in ('umi','uma','rma'):
        m=ExperimentModel(replace(ref,scenario=scenario)); f=m.make_fixture(0)
        for csi in ('perfect','ls_nn','ls_lin','controlled_nmse'):
            c=replace(ref,scenario=scenario,csi_mode=csi); mm=ExperimentModel(c)
            r=mm.run_packet(15,fixture=f)
            check(f'channel_{scenario}_{csi}',0<=r['ber']<=1 and r['total_ms']>0,ber=r['ber'],nmse=r['measured_csi_nmse'])
        again=m.make_fixture(0)
        check(f'fixture_repeat_{scenario}',torch.equal(f['h'],again['h']) and torch.equal(f['bits'],again['bits']))
    # Ordered SIC, soft outputs and explicit hard decisions, on fixed channels.
    from sionna.phy.mimo import lmmse_equalizer
    m=ExperimentModel(replace(ref,num_users=2,num_bs_ant=4,bits_per_symbol=2))
    g=generator(1984,device); b=torch.randint(0,2,(16000,4),generator=g,device=device).double();x=m.mapper(b)
    for strong in (False,True):
        H=torch.tensor([[1.,.9 if strong else 0.],[0.,.3 if strong else 1.],[0.,0.],[0.,0.]],device=device,dtype=torch.complex128).expand(len(x),4,2)
        no=.04; y=(H@x.unsqueeze(-1)).squeeze(-1)+no**.5*complex_noise((len(x),4),g,device)
        lin,nl=lmmse_equalizer(y,H,torch.eye(4,device=device,dtype=torch.complex128)*no)
        sic,ns=mmse_sic(y,H,torch.full((len(x),4),no,device=device),m.mapper.constellation.points)
        le=(m.demapper(lin,nl)>0)!=b;se=(m.demapper(sic,ns)>0)!=b
        el=float(le.double().mean());es=float(se.double().mean())
        lb=float(le.any(-1).double().mean());sb=float(se.any(-1).double().mean())
        check('sic_strong_interference' if strong else 'sic_low_interference',(es<el and sb<lb) if strong else abs(es-el)<.002,lmmse_ber=el,sic_ber=es,lmmse_bler=lb,sic_bler=sb,shape=list(sic.shape))
    report={'passed':all(c['passed'] for c in checks),'source_hash':fingerprint(),'checks':checks,'device':device}
    if output: Path(output).write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='validation.json');parser.add_argument('--device',default='cpu',help='Explicit device: cpu or cuda:0');args=parser.parse_args()
    report=validate(output=args.output,device=args.device)
    raise SystemExit(0 if report['passed'] else 1)
