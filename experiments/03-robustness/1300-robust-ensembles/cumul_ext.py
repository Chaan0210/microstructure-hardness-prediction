"""Re-extract the round2 label-blind cumulant measurements for train AND test with the identical measure() (copied verbatim from
../rmse-research-20261015/round2/cumulants.py): inputs = raw image, U-Net z1 maps (push-400/unet4maps-z1-<split>.npz), noise = 'noise' column x255.
Checks train against cumulants-train.npz. -> cumul-<split>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
SCALES=(1.,2.,4.)
PGRID=np.linspace(.005,.495,2001)
U=PGRID*(1-PGRID)
RATIO=(1-6*U)/(U**(1/3)*(1-4*U)**(2/3))
def measure(x,maps,noise):
    p0=np.repeat(np.repeat(maps[0].astype(float),2,0),2,1)
    grain=np.repeat(np.repeat((maps[0]+maps[1]).astype(float),2,0),2,1)
    shadeweight=np.clip(p0,0,1)**4
    grainweight=np.clip(grain,0,1)**2
    shade=ndi.gaussian_filter(x*shadeweight,48)/np.maximum(ndi.gaussian_filter(shadeweight,48),1e-6)
    core=(slice(20,-20),slice(20,-20));w=grainweight[core];w=w/max(w.sum(),1e-8)
    reference=max(float(np.sum(w*shade[core])),1.)
    out=[]
    for sigma in SCALES:
        signal=(ndi.gaussian_filter(x,sigma)-shade)[core]
        t=signal-np.sum(w*signal)
        m2=float(np.sum(w*t*t));k3=float(np.sum(w*t**3));k4=float(np.sum(w*t**4)-3*m2*m2)
        radius=int(4*sigma+.5);axis=np.arange(-radius,radius+1)
        kernel=np.exp(-.5*(axis/sigma)**2);kernel/=kernel.sum()
        noisevar=noise**2*(np.sum(kernel**2)**2)
        signalvar=max(m2-noisevar,1e-6)
        skew=k3/signalvar**1.5;kurt=k4/signalvar**2
        p3=.5*(1+skew/np.sqrt(skew*skew+4))
        ratio=k4/max(abs(k3),1e-9)**(4/3)
        small=float(np.interp(ratio,RATIO[::-1],PGRID[::-1]));p34=small if k3<0 else 1-small
        contrast=(abs(k3)/max(p34*(1-p34)*abs(1-2*p34),1e-8))**(1/3)
        out.extend([p3,p34,np.arcsinh(skew),np.arcsinh(kurt),contrast/reference,np.sqrt(signalvar)/reference])
    return np.array(out)
P = Path(_repo_io('../..'))
for sp in ('train', 'test'):
    F = np.load(_repo_io(f'../push-800/feat-u1dmavgH2Z-anghcor-{sp}.npz'), allow_pickle=True); k = [str(x) for x in F['keys']]; ids = [str(i) for i in F['ids']]
    noise = np.nan_to_num(F['X'][:, k.index('noise')].astype(float) * 255, nan=0)
    u = np.load(_repo_io(f'../push-400/unet4maps-z1-{sp}.npz')); ix = {str(i): j for j, i in enumerate(u['ids'])}
    Z = np.array([measure(np.asarray(Image.open(_repo_path(P, f'data/{sp}/{i}.png')), float), u['maps'][ix[i]], noise[j]) for j, i in enumerate(ids)])
    np.savez(_repo_io(f'cumul-{sp}.npz'), ids=np.array(ids), X=Z); print(sp, Z.shape, np.isfinite(Z).all())
R = np.load(_repo_io('../rmse-research-20261015/round2/cumulants-train.npz'), allow_pickle=True); T = np.load(_repo_io('cumul-train.npz'))
assert [str(i) for i in R['ids']] == [str(i) for i in T['ids']]; print('train reproduction max|diff|', np.abs(R['X'] - T['X']).max())
