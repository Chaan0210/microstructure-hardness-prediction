"""Small covariance-pooling CNN with optional fold-normalized morphology input.

Original project implementation; no external model, weights or training data.
Uses existing TinyCNN trunk. Covariance is low-rank, symmetric, trace-normalized
with a fixed epsilon; no matrix inverse/eigendecomposition is required.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import math
import torch
from torch import nn
from cnn_controls import TinyCNN

def covariance_descriptor(z, rows, cols, weights):
    flat=z.flatten(2)
    centered=flat-flat.mean(dim=2,keepdim=True)
    cov=torch.bmm(centered,centered.transpose(1,2))/max(flat.shape[2]-1,1)
    mean_variance=cov.diagonal(dim1=1,dim2=2).mean(1)[:,None,None]
    normalized=cov/(mean_variance+1e-4)
    return normalized[:,rows,cols]*weights/math.sqrt(z.shape[1])

class CovarianceCNN(TinyCNN):
    def __init__(self, morphology=False, rank=12):
        super().__init__()
        self.use_morphology=morphology
        self.projection=nn.Conv2d(96,rank,1,bias=False)
        indices=torch.triu_indices(rank,rank)
        self.register_buffer('cov_rows',indices[0],persistent=False)
        self.register_buffer('cov_cols',indices[1],persistent=False)
        weights=torch.where(indices[0]==indices[1],1.,math.sqrt(2.))
        self.register_buffer('cov_weights',weights,persistent=False)
        width=194+rank*(rank+1)//2+(26 if morphology else 0)
        self.head=nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1))

    def forward(self,x,morphology=None):
        h=self.features(x)
        stats=[h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt(),
               x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt(),
               covariance_descriptor(self.projection(h),self.cov_rows,self.cov_cols,self.cov_weights)]
        if self.use_morphology:
            if morphology is None or morphology.shape!=(x.shape[0],26):
                raise ValueError('Expected26 fold-normalized morphology features')
            stats.append(morphology)
        return self.head(torch.cat(stats,dim=1)).squeeze(1)
