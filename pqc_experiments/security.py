"""GGH research primitives. Basis quality is not a cryptographic security proof."""
from dataclasses import dataclass, asdict
import hashlib
import time
import torch
from metrics import generate_good_private_basis, hadamard_ratio

@dataclass(frozen=True)
class GGHConfig:
    rounds: int = 5
    max_coeff: int = 3
    pert: int = 2
    good_hr: float = .8
    sigma_fraction: float = .75

@dataclass(frozen=True)
class GGHKey:
    n: int
    seed: int
    parameters: GGHConfig
    R: torch.Tensor
    B: torch.Tensor
    U: torch.Tensor
    R_inv: torch.Tensor
    U_inv: torch.Tensor
    sigma: float
    metadata: dict

    @classmethod
    def generate(cls, n, seed=20, parameters=GGHConfig()):
        if n < 2 or parameters.rounds < 1 or parameters.max_coeff < 1 or parameters.pert < 1 or not 0 < parameters.sigma_fraction < 1: raise ValueError('Invalid GGH parameters')
        start=time.perf_counter()
        # CPU setup is deterministic and leaves the communication RNG untouched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            R=generate_good_private_basis(n,pert=parameters.pert,goodTh=parameters.good_hr)
            U=torch.eye(n,dtype=torch.float64); Ui=U.clone()
            # Track inverse elementary operations exactly, rather than pinv(B).
            for _ in range(parameters.rounds):
                perm=torch.randperm(n)
                for i in range(n):
                    j=int(perm[i]); j=(j+1)%n if i==j else j
                    a=int(torch.randint(1,parameters.max_coeff+1,(1,))) * (1 if torch.rand(())>.5 else -1)
                    U[i]+=a*U[j]; Ui[:,j]-=a*Ui[:,i]
            if max(U.abs().max(),Ui.abs().max()) >= 2**50: raise ArithmeticError('Unimodular coefficients exceed reliable float64 range')
            if not torch.equal(U@Ui,torch.eye(n,dtype=torch.float64)): raise ArithmeticError('Unimodular inverse lost exactness; reject key')
            Ri=torch.linalg.inv(R); B=R@U
            rho=float(Ri.abs().sum(-1).max()); sigma=parameters.sigma_fraction/(2*rho)
            meta={'hr_R':float(hadamard_ratio(R)), 'hr_B':float(hadamard_ratio(B)), 'rho_R':rho,'sigma':sigma,
                  'B_bytes':B.numel()*B.element_size(),'R_bytes':R.numel()*R.element_size(),
                  'U_bytes':U.numel()*U.element_size(),'U_inv_bytes':Ui.numel()*Ui.element_size(),
                  'public_key_bytes':B.numel()*B.element_size(), 'private_key_bytes':(R.numel()+Ui.numel())*8,
                  'intermediate_matrix_bytes':(Ri.numel()+U.numel())*8}
            meta['key_id']=hashlib.sha256(B.numpy().tobytes()+repr(asdict(parameters)).encode()).hexdigest()[:20]
            meta['keygen_ms']=(time.perf_counter()-start)*1000
            return cls(n,seed,parameters,R,B,U,Ri,Ui,sigma,meta)

class KeyRegistry:
    def __init__(self): self.keys={}
    def get(self,n,seed=20,parameters=GGHConfig()):
        identity=(n,seed,parameters)
        if identity not in self.keys: self.keys[identity]=GGHKey.generate(*identity)
        return self.keys[identity]

class TileLayout:
    def __init__(self,c): self.c=c
    def pack(self,x):
        c=self.c; lead=x.shape[:-1]
        return x.reshape(*lead,c.num_data_intervals//c.tile_time,c.tile_time,c.fft_size//c.tile_width,c.tile_width).transpose(-3,-2).reshape(*lead,-1,c.lattice_n)
    def unpack(self,x):
        c=self.c; lead=x.shape[:-2]
        return x.reshape(*lead,c.num_data_intervals//c.tile_time,c.fft_size//c.tile_width,c.tile_time,c.tile_width).transpose(-3,-2).reshape(*lead,-1)

class GGHCodec:
    def __init__(self,c,key):
        if key.n != c.lattice_n: raise ValueError('Wrong key dimension')
        self.c=c; self.key=key; self.layout=TileLayout(c)
        self.B=key.B.to(device=c.device,dtype=torch.complex128)
        self.Ri=key.R_inv.to(device=c.device,dtype=torch.complex128)
        self.Ui=key.U_inv.to(device=c.device,dtype=torch.complex128)
        self.U=key.U.to(device=c.device,dtype=torch.complex128)
        self.qam_scale=(2*(2**c.bits_per_symbol-1)/3)**.5
    def encrypt(self,x,generator):
        m=self.layout.pack(x*self.qam_scale)
        e=(2*torch.randint(0,2,m.shape,generator=generator,device=x.device)-1)*self.key.sigma
        cipher=m@self.B.T+e
        pc=cipher.abs().square().mean((-2,-1),keepdim=True)
        pm=m.abs().square().mean((-2,-1),keepdim=True)
        scale=pc.sqrt() if self.c.true_power_normalization else torch.ones_like(pc)
        return self.layout.unpack(cipher/scale), {'scale':scale,'cipher_power':pc,'message_power':pm,'message':m}
    def recover(self,x,state):
        z=(self.layout.pack(x)*state['scale'])@self.Ri.T
        z=torch.complex(z.real.round(),z.imag.round())
        m=z@self.Ui.T
        return self.layout.unpack(m/self.qam_scale), (z,m)
    def diagnostics(self,coordinates,state):
        z,m=coordinates
        return {'wrong_babai_coordinates':int((abs(z-state['message']@self.U.T)>.1).sum()),'wrong_message_symbols':int((abs(m-state['message'])>.1).sum())}

    def decrypt(self,x,state):
        x,coordinates=self.recover(x,state)
        return x,self.diagnostics(coordinates,state)

def hard_to_soft(demapper,x,crossover):
    """Fixed BSC surrogate: log P(bit=1)/P(bit=0); no oracle error calibration."""
    import math
    points=demapper.constellation.points
    sliced=points[(x.unsqueeze(-1)-points).abs().argmin(-1)]
    hard=demapper(sliced,torch.full_like(x.real,1e-6))>0
    return (2*hard.to(torch.float64)-1)*math.log((1-crossover)/crossover)

def soft_ggh_receiver(*args,**kwargs):
    raise NotImplementedError('Extension point: propagate lattice candidate likelihoods before Babai rounding')
