"""Raw-first experiments. Timing observations are batch wall times, not kernel times."""
from dataclasses import asdict,replace
from pathlib import Path
import csv
import hashlib
import json
import platform
import statistics
import time
import torch
import sionna
from .config import ExperimentConfig
from .model import ExperimentModel
from .security import KeyRegistry

def paper_groups(ref=ExperimentConfig()):
    plain=replace(ref,security_mode='none',coded=False,coderate=1.)
    full=replace(plain,security_mode='ggh_full',tile_width=ref.fft_size,tile_time=ref.lattice_n//ref.fft_size)
    sub=replace(full,security_mode='ggh_subband',tile_width=ref.fft_size//2,tile_time=2*ref.lattice_n//ref.fft_size)
    coded=lambda c:replace(c,coded=True,coderate=.5)
    # Four PHYs in every factor group: plain, plain+LDPC, GGH (Babai), GGH+LDPC.
    phys=(plain,coded(plain),full,coded(full))
    csi=[dict(csi_mode=v) for v in ('perfect','ls_nn','ls_lin')]+[dict(csi_mode='controlled_nmse',csi_nmse_db=v) for v in (-20.,-60.,-100.,-140.)]
    hardware=[{},dict(tx_evm_db=-25.),dict(tx_evm_db=-35.),dict(tx_evm_db=-45.),dict(adc_bits=10),dict(adc_bits=12),dict(adc_bits=16)]
    return {
        '01_conventional_coding':[plain,coded(plain)],
        '02_full_ggh':[plain,full],
        '03_ggh_coding':[full,coded(full)],
        '04_localization':[full,sub,coded(full),coded(sub)],
        '05_receiver':[replace(c,receiver=v) for c in phys for v in ('zf','lmmse','mmse_sic')],
        '06_csi':[replace(c,**v) for c in phys for v in csi],
        '07a_users':[replace(c,num_users=v) for c in phys for v in (1,2,4,8)],
        '07b_antennas':[replace(c,num_bs_ant=v) for c in phys for v in (4,8,16,32)],
        '08_modulation':[replace(c,bits_per_symbol=v) for c in phys for v in (2,4,6,8)],
        '09_channel':[replace(c,scenario=v) for c in phys for v in ('umi','uma','rma')],
        '10_overhead':[plain,coded(plain),full,coded(full),sub,coded(sub)],
        '11_hardware':[replace(c,**v) for c in phys for v in hardware],
    }

def ebno_for(c,ebno_values):
    """A dict gives separate grids, {'plain':[...],'ggh':[...]}; GGH needs about gamma dB more."""
    return ebno_values['plain' if c.security_mode=='none' else 'ggh'] if isinstance(ebno_values,dict) else ebno_values

def fingerprint():
    root=Path(__file__).parent
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted(root.glob('*.py')))+ (root.parent/'metrics.py').read_bytes()).hexdigest()

def run_group(configs,ebno_values,output,validation,registry=None,packets=20,warmups=2):
    if not validation.get('passed') or validation.get('source_hash')!=fingerprint(): raise RuntimeError('Run validation against this source before sweeps')
    if warmups<1 or packets<1: raise ValueError('At least one warm-up and packet required')
    registry=registry or KeyRegistry(); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    manifest={'configs':[asdict(c) for c in configs],'ebno_db':list(ebno_values),'packets':packets,'warmups':warmups,'source_hash':fingerprint(),'python':platform.python_version(),'torch':torch.__version__,'sionna':sionna.__version__,'validation':validation,'hardware':platform.platform()+' '+platform.machine(),'torch_threads':torch.get_num_threads(),'cuda':torch.version.cuda,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    with (output/'raw.jsonl').open('w') as raw:
        for index,c in enumerate(configs):
            key=registry.get(c.lattice_n,c.seed) if c.security_mode!='none' else None
            if key is not None:
                key_record={'dimension':key.n,'seed':key.seed,'parameters':asdict(key.parameters),'measurements':key.metadata}
                (output/f"key_{key.metadata['key_id']}.json").write_text(json.dumps(key_record,indent=2))
            model=ExperimentModel(c,key); grid=ebno_for(c,ebno_values)
            for warm in range(warmups): model.run_packet(grid[0],packet=100000+warm)
            for ebno in grid:
                for packet in range(packets):
                    row=model.run_packet(ebno,packet); row['config_index']=index
                    raw.write(json.dumps(row)+'\n'); raw.flush(); rows.append(row)
    summary=summarize(rows)
    write_csv(output/'summary.csv',summary)
    write_csv(output/'raw.csv',rows)
    return rows,summary

def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def summarize(rows):
    groups={}
    for r in rows: groups.setdefault((r['config_index'],r['ebno_db']),[]).append(r)
    results=[]
    for _,items in groups.items():
        r=items[0]; result={k:r[k] for k in ('config_index','scheme','scenario','K','M','modulation','receiver','csi','csi_nmse_db','coderate','lattice_n','tile','tx_evm_db','adc_bits','ebno_db','key_id')}
        for error,total,metric in [('bit_errors','num_bits','ber'),('block_errors','num_blocks','bler')]:
            n=sum(x[total] for x in items); e=sum(x[error] for x in items); result[metric]=e/n
            # Wilson interval; block intervals are the primary reliability report.
            z=1.96;p=e/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
            result[metric+'_ci_low']=max(0,mid-half);result[metric+'_ci_high']=min(1,mid+half)
            result[error]=e; result[total]=n
        for k in r:
            values=[x[k] for x in items if isinstance(x.get(k),(int,float)) and not isinstance(x[k],bool)]
            if k.endswith('_ms') or k.endswith('_ms_per_block'):
                ordered=sorted(values);result[k]=statistics.mean(values);result[k+'_std']=statistics.pstdev(values);result[k+'_median']=statistics.median(values);result[k+'_p95']=ordered[min(len(ordered)-1,int(.95*len(ordered)))]
            elif k in ('goodput','effective_spectral_efficiency','post_equalization_sinr','cipher_power','cipher_power_db','gamma','gamma_db','tx_data_power','tx_grid_power','measured_csi_nmse','hr_R','hr_B','rho_R','sigma','public_key_bytes','private_key_bytes','intermediate_matrix_bytes','security_overhead_bits_per_frame','normalization_sideinfo_bits_per_frame','runtime_key_matrix_bytes','ldpc_iterations_executed'):
                result[k]=statistics.mean(values) if values else None
            elif k in ('wrong_babai_coordinates','wrong_message_symbols'): result[k]=sum(values)
        result['retransmission_probability']=result['bler'];results.append(result)
    for result in results:
        curve=sorted((r for r in results if r['config_index']==result['config_index']),key=lambda r:r['ebno_db'])
        threshold=None;status='not_reached'
        if curve[0]['bler']<=.01: status='below_sweep_range'
        else:
            for a,b in zip(curve,curve[1:]):
                if a['bler']>.01 and b['bler']<=.01:
                    # Linear probability interpolation, descriptive only, no extrapolation.
                    threshold=a['ebno_db']+(.01-a['bler'])/(b['bler']-a['bler'])*(b['ebno_db']-a['ebno_db']);status='bracketed';break
        result['required_ebno_bler_1e2']=threshold;result['threshold_status']=status
    return results
