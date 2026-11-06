"""Original project control: reduced covariance capacity and bounded normalization."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import math
import torch
from torch import nn
from cnn_texture_fusion import CovarianceCNN

def regularized_descriptor(z,rows,cols,weights):
    flat=z.flatten(2);centered=flat-flat.mean(2,keepdim=True)
    cov=torch.bmm(centered,centered.transpose(1,2))/max(flat.shape[2]-1,1)
    diagonal=cov.diagonal(dim1=1,dim2=2)
    # PSD diagonal shrinkage; per-channel normalization includes relative+absolute floors.
    shrunk=.5*cov+.5*torch.diag_embed(diagonal)
    denominator=(diagonal+.1*diagonal.mean(1,keepdim=True)+1e-4).sqrt()
    normalized=shrunk/denominator[:,:,None]/denominator[:,None,:]
    return normalized[:,rows,cols]*weights/math.sqrt(z.shape[1])

class SmallCovarianceCNN(CovarianceCNN):
    def __init__(self,regularized=False):
        super().__init__(morphology=False,rank=6)
        self.regularized=regularized
        self.head=nn.Sequential(nn.Linear(194+21,32),nn.SiLU(),nn.Dropout(.15),nn.Linear(32,1))
    def forward(self,x):
        h=self.features(x)
        z=self.projection(h)
        if self.regularized:
            cov=regularized_descriptor(z,self.cov_rows,self.cov_cols,self.cov_weights)
        else:
            from cnn_texture_fusion import covariance_descriptor
            cov=covariance_descriptor(z,self.cov_rows,self.cov_cols,self.cov_weights)
        stats=[h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt(),x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt(),cov]
        return self.head(torch.cat(stats,1)).squeeze(1)
