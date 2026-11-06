"""Bounded joint inverse-rendering feasibility model, not a generator recovery.
Own phase-field/sparse-dark-layer approximation; not Potts or VB reproduction.
No labels, file access, or learned shared image parameters in this module.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import h_minima, disk
from skimage.segmentation import watershed
import denoised_grain_features as old
import phase_features as phase

STEPS=100
START_SIGMAS=(0.5,1.2)
NAMES=['grain_diameter','dark_phase_fraction','pore_fraction','alignment']

def logit(x):
    return torch.logit(torch.clamp(x,0.001,0.999))

def summarize(u,d):
    dark=d>0.25
    pore=ndi.binary_opening(dark,structure=disk(2))
    boundary=dark & ~pore
    grad=ndi.gaussian_gradient_magnitude(u,0.7)
    top=d+0.25*grad/max(float(np.percentile(grad,90)),1e-6)
    mins=h_minima(top,max(0.1*float(np.percentile(top,90)),1e-5)) & ~dark
    markers=label(mins)
    if not markers.any(): markers.flat[int(np.argmin(top))]=1
    seg=watershed(top,markers,watershed_line=True)
    s=old._summarize(seg,u)
    f=phase._fit_populations(u[~dark])
    return np.array([s[2],f[0],pore.mean(),s[7]]),boundary,pore,seg

def estimate(image,device='cpu'):
    x=np.asarray(image,dtype=np.float32)/255
    lo,hi=np.percentile(x,(2,98)); span=max(float(hi-lo),0.05)
    z=(x-lo)/span
    smooth=ndi.gaussian_filter(z,0.6)
    closing=ndi.grey_closing(smooth,size=(7,7),mode='reflect')
    dark0=np.clip((closing-smooth)/0.35,0.003,0.97)
    region0=np.clip(smooth+0.35*dark0,0.003,0.997)
    n0=float(np.clip(old._noise_sigma(x)/span,0.015,0.5))
    b=len(START_SIGMAS)
    def tensor(v): return torch.as_tensor(v,dtype=torch.float32,device=device)
    target=tensor(z)[None,None].repeat(b,1,1,1)
    ur=torch.nn.Parameter(logit(tensor(region0)[None,None].repeat(b,1,1,1)))
    dr=torch.nn.Parameter(logit(tensor(dark0)[None,None].repeat(b,1,1,1)))
    sr=torch.nn.Parameter(logit(tensor([(v-0.25)/1.5 for v in START_SIGMAS])))
    ar=torch.nn.Parameter(torch.zeros(b,device=device))
    br=torch.nn.Parameter(torch.zeros(b,device=device))
    nr=torch.nn.Parameter(torch.zeros(b,device=device))
    rr=torch.nn.Parameter(torch.zeros(b,device=device))
    pars=[ur,dr,sr,ar,br,nr,rr]
    opt=torch.optim.Adam(pars,lr=0.04)
    coord=torch.arange(-6,7,device=device,dtype=torch.float32)
    losses=[]
    def forward():
        u=torch.sigmoid(ur); d=torch.sigmoid(dr)
        sigma=0.25+1.5*torch.sigmoid(sr)
        k=torch.exp(-0.5*(coord[None,:]/sigma[:,None])**2); k=k/k.sum(1,keepdim=True)
        latent=u-0.35*d
        v=latent.transpose(0,1)
        v=F.conv2d(F.pad(v,(6,6,0,0),mode='reflect'),k[:,None,None,:],groups=b)
        v=F.conv2d(F.pad(v,(0,0,6,6),mode='reflect'),k[:,None,:,None],groups=b).transpose(0,1)
        a=0.5+torch.sigmoid(ar); off=0.25*torch.tanh(br)
        pred=a[:,None,None,None]*v+off[:,None,None,None]
        noise=n0*torch.exp(0.69314718*torch.tanh(nr))
        rho=0.65*torch.tanh(rr)
        e=target-pred; r=rho[:,None,None,None]
        w=e[:,:,1:,1:]-r*(e[:,:,:-1,1:]+e[:,:,1:,:-1])+r*r*e[:,:,:-1,:-1]
        data=(w.square().mean((1,2,3))/noise.square()+2*torch.log(noise)).mean()
        # Approximate piecewise-constant region prior; no fixed tessellation.
        dx=u[:,:,:,1:]-u[:,:,:,:-1]; dy=u[:,:,1:,:]-u[:,:,:-1,:]
        jumps=(torch.log1p((dx/0.035).square()).mean()+torch.log1p((dy/0.035).square()).mean())/2
        dtv=((d[:,:,:,1:]-d[:,:,:,:-1]).abs().mean()+(d[:,:,1:,:]-d[:,:,:-1,:]).abs().mean())/2
        # Sparse dark layer accommodates thin boundaries and compact pores.
        # No pore truth or region count is given to the optimizer.
        prior=0.18*jumps+0.12*d.mean()+0.05*dtv
        prior=prior+0.025*(ar.square().mean()+br.square().mean())+0.01*rho.square().mean()
        return data+prior,(u,d,pred,sigma,a,off,noise,rho)
    for step in range(STEPS):
        opt.zero_grad(); loss,_=forward(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pars,10);opt.step()
        if step in (0,49,99):losses.append(float(loss.detach().cpu()))
    with torch.no_grad(): loss,states=forward()
    u,d,pred,sig,a,off,n,rho=[v.detach().cpu().numpy() for v in states]
    vals=[]; boundaries=[];pores=[];segs=[]
    for j in range(b):
        f,bd,p,sg=summarize(u[j,0],d[j,0]);vals.append(f);boundaries.append(bd);pores.append(p);segs.append(sg)
    vals=np.asarray(vals)
    diag={'loss_samples':losses,'blur_sigma':sig.tolist(),'contrast':a.tolist(),'offset':off.tolist(),'noise_normalized':n.tolist(),'rho':rho.tolist(),'noise_initial_normalized':n0,'data_rmse_normalized':np.sqrt(((pred-target.detach().cpu().numpy())**2).mean((1,2,3))).tolist(),'latent_std':u.std((1,2,3)).tolist(),'boundary_fraction':[float(v.mean()) for v in boundaries],'pore_fraction':[float(v.mean()) for v in pores],'start_feature_range':np.ptp(vals,axis=0).tolist()}
    # Neither initial solution is selected from labels or lower objective.
    return {'features':np.median(vals,axis=0),'start_features':vals,'boundary':np.mean(boundaries,axis=0)>=0.5,'pore':np.mean(pores,axis=0)>=0.5,'u':u[:,0].mean(0),'d':d[:,0].mean(0),'render':(pred[:,0].mean(0)*span+lo),'diagnostics':diag}

def baseline(image):
    g,_,_=old.extract_denoised(image); p,_=phase.extract_phase(image)
    x=ndi.gaussian_filter(image.astype(float)/255,1,mode='reflect')
    low,high=np.percentile(x,(2,98));pores=np.zeros_like(x,dtype=bool)
    for reg in regionprops(label(x<low-0.1*(high-low))):
        if 2<=reg.area<=128 and reg.eccentricity<0.9 and reg.solidity>0.8:
            sl=tuple(slice(a,b) for a,b in zip(reg.bbox[:2],reg.bbox[2:]));pores[sl]|=reg.image
    pores=ndi.binary_dilation(pores,structure=np.ones((3,3),dtype=bool))
    ridge=ndi.grey_closing(x,size=(7,7),mode='reflect')-x
    boundary=ridge>threshold_otsu(ridge) if np.ptp(ridge)>1e-10 else np.zeros_like(pores)
    return {'features':np.array([g[4],p[10],p[18],g[14]]),'boundary':boundary,'pore':pores}
