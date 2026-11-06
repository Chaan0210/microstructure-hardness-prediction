"""Frozen final DINO patch-channel covariance after fixed data-blind projection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
SEED=20261225
CHANNELS=384
RANK=32
ROWS,COLS=np.triu_indices(RANK)
NAMES=tuple(f'final12_projected_cov_{a}_{b}' for a,b in zip(ROWS,COLS))
def projection():
 # Each projected coordinate averages channel energy in expectation. This is
 # not JL norm-preserving scaling, PCA, learned projection, or whitening.
 return np.random.default_rng(SEED).standard_normal((CHANNELS,RANK))/np.sqrt(CHANNELS)
def describe(tokens,R):
 P=np.asarray(tokens,dtype=np.float64)
 assert P.ndim==2 and P.shape[1]==CHANNELS and P.shape[0]>1 and np.isfinite(P).all()
 centered=P-P.mean(0);Z=centered@R;C=Z.T@Z/len(Z);C=(C+C.T)*.5
 diagonal_surrogate=R.T@((centered**2).mean(0)[:,None]*R)
 values=C[ROWS,COLS];assert values.shape==(528,) and np.isfinite(values).all()
 return values,C,diagonal_surrogate
