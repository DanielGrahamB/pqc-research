import torch

def mmse_sic(y,h,noise,points=None):
    """Batched ordered per-RE SIC: y [..., M], H [..., M, K].

    Plain PHY uses nearest-QAM hard cancellation. Ciphertext uses continuous
    cancellation (no independent QAM alphabet). Reliability assumes previous
    cancellations were correct; propagation of their errors is not modeled.
    """
    K=h.shape[-1]; M=h.shape[-2]; shape=h.shape[:-2]
    residual=y.clone(); active=h.clone()
    ids=torch.arange(K,device=h.device).expand(*shape,K).clone()
    estimates=torch.zeros(*shape,K,dtype=h.dtype,device=h.device)
    variances=torch.zeros(*shape,K,dtype=h.real.dtype,device=h.device)
    nv=torch.broadcast_to(noise,(*shape,M)).clamp_min(1e-12)
    for _ in range(K):
        cov=active@active.mH+torch.diag_embed(nv).to(h.dtype)
        w=torch.linalg.solve(cov,active).mH
        gain=(w@active).diagonal(dim1=-2,dim2=-1).real.clamp_min(1e-12)
        ne=(1/gain-1).clamp_min(1e-12)
        chosen=ne.argmin(-1,keepdim=True)
        soft=(w@residual.unsqueeze(-1)).squeeze(-1)/gain
        value=soft.gather(-1,chosen)
        variance=ne.gather(-1,chosen)
        original=ids.gather(-1,chosen)
        estimates.scatter_(-1,original,value); variances.scatter_(-1,original,variance)
        if points is None: decision=value
        else:
            alphabet=points if points.ndim==1 else torch.broadcast_to(points,(*shape,K,points.shape[-1])).gather(-2,original.unsqueeze(-1).expand(*shape,1,points.shape[-1])).squeeze(-2)
            selected=(value-alphabet).abs().argmin(-1,keepdim=True)
            decision=alphabet[selected] if alphabet.ndim==1 else alphabet.gather(-1,selected)
        column=active.gather(-1,chosen.unsqueeze(-2).expand(*shape,M,1)).squeeze(-1)
        residual=residual-column*decision
        if active.shape[-1]>1:
            keep=torch.arange(active.shape[-1],device=h.device).expand_as(ids)!=chosen
            active=active.transpose(-2,-1)[keep].reshape(*shape,-1,M).transpose(-2,-1)
            ids=ids[keep].reshape(*shape,-1)
    return estimates,variances
