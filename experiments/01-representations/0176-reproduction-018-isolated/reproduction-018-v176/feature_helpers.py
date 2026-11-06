from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np,torch
from torch import nn
def dino_input(path):
 with Image.open(_repo_io(path)) as im:
  assert im.size==(256,256) and im.mode=='L';a=np.asarray(im.resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/255.
 mean=np.array([.485,.456,.406],np.float32)[:,None,None];std=np.array([.229,.224,.225],np.float32)[:,None,None]
 return np.ascontiguousarray((np.repeat(a[None],3,axis=0)-mean)/std)

class ShortHead(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(6912,32),nn.GELU(),nn.Dropout(.5),nn.Linear(32,1))
 def forward(self,x):return self.net(x).squeeze(-1)

class Short(nn.Module):
 def __init__(self,mean,scale,seed=20270215):
  super().__init__();torch.manual_seed(seed);self.head=ShortHead();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
 def forward(self,x):return self.head((x-self.mean)/self.scale)

def spatial_descriptor(tokens):
 global SPATIAL_PROJECTION
 if SPATIAL_PROJECTION is None:SPATIAL_PROJECTION=np.linalg.qr(np.random.default_rng(20271201).normal(size=(768,32)),mode='reduced')[0]
 z=(tokens.astype(float)@SPATIAL_PROJECTION).reshape(36,36,32);h=np.diff(z,axis=0).reshape(-1,32);v=np.diff(z,axis=1).reshape(-1,32);mat=(h.T@h+v.T@v)/(2*(len(h)+len(v)));return mat[np.triu_indices(32)]
SPATIAL_PROJECTION=None
from PIL import Image
