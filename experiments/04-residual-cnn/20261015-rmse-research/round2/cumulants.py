"""Label-blind shade/pore-conditioned third and fourth cumulant measurements."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from models import HERE,inputs
import research as r

SCALES=(1.,2.,4.)
PGRID=np.linspace(.005,.495,2001)
U=PGRID*(1-PGRID)
RATIO=(1-6*U)/(U**(1/3)*(1-4*U)**(2/3))
assert np.all(np.diff(RATIO)<0)


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
        # Finite discrete Gaussian kernel, excluding boundary reflection effects.
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


def extract():
    d=inputs();path=r.EXP/'push-400/unet4maps-z1-train.npz';u=np.load(_repo_io(path))
    ix={str(i):j for j,i in enumerate(u['ids'])};maps=u['maps'];out=[];hashes={};start=time.monotonic()
    for j,i in enumerate(d['ids']):
        p=_repo_path(r.ROOT, f'data/train/{i}.png');x=np.asarray(Image.open(_repo_io(p)),float)
        hashes[str(i)]=r.digest(p)
        out.append(measure(x,maps[ix[str(i)]],d['noise'][j]))
        if (j+1)%100==0:print('features',j+1,'seconds',round(time.monotonic()-start,1),flush=True)
    keys=[f'cumulant_sigma{s}_{name}' for s in SCALES for name in ('p3','p34','asinh_skew','asinh_kurt','contrast','signal_sd')]
    z=np.array(out);assert z.shape==(500,18) and np.isfinite(z).all()
    np.savez_compressed(_repo_io(HERE/'cumulants-train.npz'),ids=d['ids'],X=z,keys=np.array(keys))
    r.save_json(HERE/'cumulants-provenance.json',dict(map_sha256=r.digest(path),image_sha256=hashes,
        code_sha256=r.digest(__file__),scales=SCALES,features=keys,hardness_used=False,seconds=time.monotonic()-start,
        caveat='Weighted spatial plug-in cumulants, not unbiased k-statistics; Gaussian independent noise and binary-mixture interpretations are approximations.'))


class ExtraRidge:
    def __init__(self,alpha,scale):self.alpha,self.scale=alpha,scale
    def fit(self,x,e,y):
        self.imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
        self.ss=StandardScaler().fit(self.imp.transform(x));z=self.ss.transform(self.imp.transform(x))
        if self.scale:
            self.es=StandardScaler().fit(e);z=np.c_[z,self.scale*self.es.transform(e)]
        self.model=Ridge(alpha=self.alpha).fit(z,y);return self
    def predict(self,x,e):
        z=self.ss.transform(self.imp.transform(x))
        if self.scale:z=np.c_[z,self.scale*self.es.transform(e)]
        return self.model.predict(z)


def select(x,e,y,seed):
    grid=[(a,s) for s in (0.,.1,1.) for a in (.1,1,10,100,1000)];loss=np.zeros(len(grid))
    for tr,va in KFold(3,shuffle=True,random_state=seed).split(x):
        for j,(a,s) in enumerate(grid):
            p=ExtraRidge(a,s).fit(x[tr],e[tr],y[tr]).predict(x[va],e[va]);loss[j]+=np.sum((p-y[va])**2)
    j0=int(np.argmin(loss[:5]));j=int(np.argmin(loss))
    return [ExtraRidge(*grid[k]).fit(x,e,y) for k in (j0,j)],dict(null=grid[j0],selected=grid[j],grid=grid,inner_mse=(loss/len(y)).tolist())


def evaluate():
    d=inputs();dev=d['dev'];y=d['y'][dev];fold=d['fold'];base=d['strict_base']
    gx=np.load(_repo_io(r.HERE/'gmm-features-dev.npz'));assert np.array_equal(gx['ids'],d['ids'][dev]);gx=gx['X']
    c=np.load(_repo_io(HERE/'cumulants-train.npz'));assert np.array_equal(c['ids'],d['ids'])
    extra=c['X'][dev];random=np.random.default_rng(r.SEED).normal(size=c['X'].shape)[dev]
    out={key:d['C73_gmm'].copy() for key in ('null','cumulants','random')};details=[];start=time.monotonic()
    for p in range(3):
        for k in range(5):
            tr=np.where(fold[p]!=k)[0]
            for noisy in (False,True):
                cols=np.ones(len(d['keys']),bool) if noisy else d['keep']
                x=np.c_[d['X'][dev][:,cols],gx];va=np.where((fold[p]==k)&((d['noise'][dev]>=12)==noisy))[0]
                seed=19001+101*p+k if noisy else 19001
                for kind,e in [('cumulants',extra),('random',random)]:
                    (null,model),meta=select(x[tr],e[tr],y[tr],seed)
                    out[kind][p,va]=model.predict(x[va],e[va]);out['null'][p,va]=null.predict(x[va],e[va])
                    meta.update(partition=p,fold=k,noisy=noisy,kind=kind);details.append(meta)
            print('cumulants',p,k,'seconds',round(time.monotonic()-start,1),flush=True)
    diff=float(np.max(np.abs(out['null']-d['C73_gmm'])[~d['fail']]))
    assert diff<1e-8,diff
    blends={key:np.where(d['fail'],base,base+.15*(pred-d['C73_gmm'])) for key,pred in out.items()}
    result={key:r.compare(base,pred,y,fold) for key,pred in blends.items()}
    result['real_vs_random']=r.compare(blends['random'],blends['cumulants'],y,fold)
    result['details']=details;result['null_max_difference']=diff;result['seconds']=time.monotonic()-start
    np.savez_compressed(_repo_io(HERE/'cumulants-eval.npz'),**out,**{k+'_blend':v for k,v in blends.items()})
    r.save_json(HERE/'cumulants-eval.json',result)
    print(json.dumps({k:{v:result[k][v]for v in ('gain','partition_gains','improved_folds','conditional_id_bootstrap_gain_ci95')}for k in out},indent=2))


def check():
    # The analytic two-level cumulant relation must invert exactly; no training data.
    errors=[]
    for p in (.05,.1,.2,.3,.4):
        k3=-p*(1-p)*(1-2*p);k4=p*(1-p)*(1-6*p*(1-p))
        inverse=np.interp(k4/abs(k3)**(4/3),RATIO[::-1],PGRID[::-1])
        errors.append(abs(inverse-p))
    assert max(errors)<1e-5
    # Linear shade/background and D4 transforms use the same isotropic operations.
    rng=np.random.default_rng(1);im=rng.normal(150,8,(256,256));m=np.zeros((8,128,128));m[0]=.8;m[1]=.15
    a=measure(im,m,8);b=measure(np.rot90(im),np.rot90(m,axes=(1,2)),8)
    assert np.max(np.abs(a-b))<1e-8
    r.save_json(HERE/'cumulants-check.json',dict(binary_inversion_maxerror=max(errors),d4_maxerror=float(np.max(np.abs(a-b)))))
    print('Cumulant inversion and D4 tests passed.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['extract','evaluate','check']);a=ap.parse_args()
    with threadpool_limits(limits=2):
        if a.action=='check':check()
        elif a.action=='extract':
            if (HERE/'cumulants-train.npz').exists():raise FileExistsError('cumulants-train.npz')
            extract()
        else:
            if (HERE/'cumulants-eval.npz').exists():raise FileExistsError('cumulants-eval.npz')
            evaluate()
