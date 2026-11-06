"""Fine-resolution original intensity plus frozen segmentation context, direct MSE."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'))
import argparse,hashlib,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import grain_attention_v2 as ga

RAW=None;NECK=None

def prepare_raw():
    path=_repo_path(HERE, 'raw128-c4.npy')
    if not path.exists():
        pixels=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(500,2,4,128,128))
        for j,im in enumerate(pixels[:,0]):
            x=im.astype(np.float32)/255;mu=x.reshape(128,2,128,2).mean((1,3));var=(x*x).reshape(128,2,128,2).mean((1,3))-mu*mu
            lo,med,hi=np.percentile(x,[5,50,95]);hh=(x[::2,::2]-x[::2,1::2]-x[1::2,::2]+x[1::2,1::2])/2
            noise=float(np.median(abs(hh))/.6745);p=np.stack([(mu-.5)*4,(mu-float(med))/max(float(hi-lo),.01),np.sqrt(np.maximum(var,0))*4,np.full_like(mu,noise*4)])
            out[j,0]=p;out[j,1]=np.rot90(p,1,axes=(1,2));
        out.flush();del out
    return np.load(_repo_io(path),mmap_mode='r')

class ImageData:
    def __init__(self,bags,weights,G,ctx):
        self.g=torch.tensor(np.clip((G-ctx['global_mean'])/ctx['global_std'],-10,10),dtype=torch.float32)
        self.rng=np.random.default_rng(2026101871);self.view=None
    def batch(self,ix):
        view=self.rng.integers(2,size=len(ix)) if self.view is None else np.full(len(ix),self.view)
        neck=torch.from_numpy(np.array(NECK[ix,view],dtype=np.float32));raw=torch.from_numpy(np.array(RAW[ix,view],dtype=np.float32))
        return neck,raw,self.g[ix]

class Fusion(nn.Module):
    def __init__(self,od,gd,ctx,kind):
        super().__init__();self.kind=kind;self.neck=nn.Conv2d(256,16,1)
        self.local=nn.Sequential(nn.Conv2d(20,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,8,3,padding=1))
        self.base=nn.Linear(gd,1);self.head=nn.Sequential(nn.Linear(40+gd,64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,1))
        with torch.no_grad():self.base.weight.copy_(torch.tensor(ctx['base_w'][None],dtype=torch.float32));self.base.bias.fill_(ctx['base_b'])
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,neck,raw,g):
        neck=torch.zeros_like(neck) if self.kind=='raw' else neck
        raw=torch.zeros_like(raw) if self.kind=='neck' else raw
        z=F.interpolate(self.neck(neck),size=(128,128),mode='bilinear',align_corners=False)
        p=self.local(torch.cat([z,raw],1)).sigmoid();stats=[p.mean((2,3)),(p.var((2,3),correction=0)+1e-6).sqrt()]
        for k in [16,32,64]:stats.append((F.avg_pool2d(p,k).var((2,3),correction=0)+1e-6).sqrt())
        return self.base(g)[:,0]+self.head(torch.cat(stats+[g],1))[:,0]

@torch.no_grad()
def predict(model,data,ix,ctx):
    model.eval();out=[]
    for first in range(0,len(ix),16):
        vals=[]
        for view in [0,1]:data.view=view;vals.append(model(*data.batch(ix[first:first+16])).numpy())
        out.extend(np.mean(vals,0)*ctx['ys']+ctx['ym'])
    data.view=None;return np.array(out)

def main(a):
    global RAW,NECK
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert np.load(_repo_path(HERE, 'cpsam-neck-c4/done.npy')).all();RAW=prepare_raw();NECK=np.load(_repo_path(HERE, 'cpsam-neck-c4/maps.npy'),mmap_mode='r')
    bags,weights,G,d,prov=ga.load();b=np.load(_repo_path(ROOT, 'experiments/probabilistic-grains-900/object-bank-train.npz'))
    ok=list(b['object_keys']);gk=list(b['global_keys']);names=[f'mean_{k}' for k in ok]+[f'sd_{k}' for k in ok]+gk
    drop={f'{s}_{k}' for s in ['mean','sd'] for k in ['cx','cy','cos2','sin2']}|{'orient_c2','orient_s2'}
    keep=np.array([n not in drop for n in names]);G=G[:,keep]
    tr=d['dev'][d['fold']!=a.fold];va=d['dev'][d['fold']==a.fold];A,B=train_test_split(tr,test_size=.2,random_state=2026101847+a.fold)
    out=_repo_path(HERE, 'runs', a.tag);out.mkdir(parents=True,exist_ok=True);assert not (out/'done.json').exists()
    ga.Model=Fusion;ga.Data=ImageData;ga.predict=predict
    ga.save(_repo_io(out/'recipe.json'),vars(a)|dict(A=A.tolist(),B=B.tolist(),T=tr.tolist(),V=va.tolist(),globals=[n for n,k in zip(names,keep) if k],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),trainer_sha256=hashlib.sha256((_repo_path(HERE, 'grain_attention_v2.py')).read_bytes()).hexdigest(),
        object_bank=prov,representation='Eligible fixed cpsam neck maps; original2x2 grayscale moments at128resolution; original/rot90, signed fourth-order orientation retained.',
        objective='Direct original-image hardness MSE, jointly trained global linear branch; no fixed residual targets or fitted predictions as inputs.'))
    r=ga.fit(bags,weights,G,d['y'],A,B,a.kind,a.seed,a.epochs,out/'selector',select=True)
    r2=ga.fit(bags,weights,G,d['y'],tr,va,a.kind,a.seed,r['selected_epoch'],out/'refit')
    p=np.load(_repo_io(out/'refit/predictions.npz'))['prediction'];base=d['base083'][d['fold']==a.fold]
    ga.save(_repo_io(out/'done.json'),dict(selector=r,refit=r2,base083_rmse=ga.rmse(base,d['y'][va]),quarter_blend_rmse=ga.rmse(.75*base+.25*p,d['y'][va])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--kind',choices=['fusion','raw','neck'],default='fusion');p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026101873);p.add_argument('--epochs',type=int,default=80)
    with threadpool_limits(limits=2):main(p.parse_args())
