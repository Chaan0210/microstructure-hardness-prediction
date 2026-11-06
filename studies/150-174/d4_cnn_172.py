"""Exact finite D4 weight-tied convolutions; no transformed-image ensemble."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import math
import torch
from torch import nn
from torch.nn import functional as F

def action(x,g):
    z=torch.rot90(x,g//2,(-2,-1))
    return z.flip(-1) if g%2 else z
# g = horizontal_flip**f after rotation**r. Build composition from a faithful action.
_fixture=torch.arange(25).reshape(5,5)
MULT=tuple(tuple(next(k for k in range(8) if torch.equal(action(action(_fixture,h),g),action(_fixture,k))) for h in range(8)) for g in range(8))
INV=tuple(next(h for h in range(8) if MULT[g][h]==0 and MULT[h][g]==0) for g in range(8))
PERM=tuple(tuple(MULT[INV[g]][h] for h in range(8)) for g in range(8))
del _fixture

def feature_action(x,g):
    return action(x[:,:,list(PERM[g])],g)

class D4Conv(nn.Module):
    def __init__(self,inc,out,k,lift=False):
        super().__init__();self.inc=inc;self.out=out;self.k=k;self.lift=lift
        self.weight=nn.Parameter(torch.empty(out,inc,1 if lift else 8,k,k));self.bias=nn.Parameter(torch.empty(out))
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5));bound=1/math.sqrt(inc*(1 if lift else 8)*k*k);nn.init.uniform_(self.bias,-bound,bound)
    def expanded(self):
        w=[action(self.weight if self.lift else self.weight[:,:,list(PERM[g])],g) for g in range(8)]
        return torch.stack(w,1).reshape(self.out*8,self.inc*(1 if self.lift else 8),self.k,self.k)
    def forward(self,x):
        if not self.lift:x=x.flatten(1,2)
        h=F.conv2d(x,self.expanded(),self.bias.repeat_interleave(8),padding=self.k//2)
        return h.reshape(h.shape[0],self.out,8,h.shape[-2],h.shape[-1])

class TiedGroupNorm(nn.Module):
    def __init__(self,c):
        super().__init__();assert c%4==0;self.weight=nn.Parameter(torch.ones(c));self.bias=nn.Parameter(torch.zeros(c))
    def forward(self,x):
        h=F.group_norm(x.flatten(1,2),4).reshape_as(x)
        return h*self.weight[None,:,None,None,None]+self.bias[None,:,None,None,None]

class D4CNN(nn.Module):
    def __init__(self):
        super().__init__();inc=1;self.convs=nn.ModuleList();self.norms=nn.ModuleList()
        for i,c in enumerate((4,12,24,36)):
            self.convs.append(D4Conv(inc,c,5 if i==0 else 3,lift=i==0));self.norms.append(TiedGroupNorm(c));inc=c
        self.head=nn.Sequential(nn.Linear(74,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1))
    def stages(self,x):
        h=x;out=[]
        for i,(conv,norm) in enumerate(zip(self.convs,self.norms)):
            if i:
                b,c,g,H,W=h.shape;h=F.avg_pool2d(h.flatten(1,2),2).reshape(b,c,g,H//2,W//2)
            else:h=F.avg_pool2d(h,2)
            h=F.silu(norm(conv(h)));out.append(h)
        return out
    def forward(self,x):
        h=self.stages(x)[-1];dims=(2,3,4)
        stats=torch.cat([h.mean(dims),(h.var(dims,correction=0)+1e-6).sqrt(),x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1)
        return self.head(stats).squeeze(1)

class PoolCNN(nn.Module):
    """Same trainable parameters/initial values as legacy TinyCNN; only downsampling changes."""
    def __init__(self):
        super().__init__();from cnn_controls import TinyCNN
        base=TinyCNN();self.features=base.features;self.head=base.head
        for layer in self.features:
            if isinstance(layer,nn.Conv2d):layer.stride=(1,1)
    def forward(self,x):
        h=x
        for i in range(0,12,3):h=self.features[i+2](self.features[i+1](self.features[i](F.avg_pool2d(h,2))))
        stats=torch.cat([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt(),x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1)
        return self.head(stats).squeeze(1)

MODELS={'group':D4CNN,'pool_control':PoolCNN}
