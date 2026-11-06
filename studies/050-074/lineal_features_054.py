"""120 image-local interpolated digital lineal-path features; no learned state.
L is pixel-center endpoint distance. All interior sampled sites must be phase.
Diagonal discrete lags have spacing sqrt(2); probabilities at adjacent integer
lags are linearly interpolated to the requested physical length. This is a
lattice approximation, not a claim about unsampled subpixel material.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
QUANTILES=(.10,.25,.50,.75,.90)
LENGTHS=(2.,4.,8.,16.,32.,64.)
DIRECTIONS=((0,1),(1,0),(1,1),(1,-1))
NAMES=tuple(f'lineal_q{int(q*100):02d}_{side}_L{int(L)}_{stat}' for q in QUANTILES for side in ('low','complement') for L in LENGTHS for stat in ('direction_mean','direction_std'))

def lines(mask):
 mask=np.asarray(mask,dtype=bool);n,m=mask.shape
 assert n==m
 yield mask,np.full(n,n),1.
 yield mask.T,np.full(n,n),1.
 xx=np.broadcast_to(np.arange(n)[None,:],(2*n-1,n));offset=np.arange(-n+1,n)[:,None]
 for yy in (xx+offset,(n-1-xx)+offset):
  valid=(yy>=0)&(yy<n);v=mask[np.clip(yy,0,n-1),xx]&valid
  yield v,valid.sum(1),np.sqrt(2.)

def run_lengths(a):
 d=np.diff(np.pad(a.astype(np.int8),((0,0),(1,1))),axis=1);r,start=np.nonzero(d==1);er,end=np.nonzero(d==-1);assert np.array_equal(r,er)
 return end-start

def discrete_probability(runs,line_sizes,k):
 # A lag k needs k+1 pixel centers; include truncated runs, but only windows
 # fully contained within the actual image. No wrap or artificial padding sites.
 denominator=int(np.maximum(line_sizes-k,0).sum())
 if denominator==0:return 0.0
 return float(np.maximum(runs-k,0).sum()/denominator)

def directional(mask,lengths=LENGTHS):
 result=[]
 for a,sizes,spacing in lines(mask):
  runs=run_lengths(a);vals=[]
  for L in lengths:
   lag=L/spacing;lo=int(np.floor(lag));hi=int(np.ceil(lag));weight=lag-lo
   first=discrete_probability(runs,sizes,lo)
   vals.append(first if lo==hi else (1-weight)*first+weight*discrete_probability(runs,sizes,hi))
  result.append(vals)
 return np.asarray(result,dtype=np.float64)

def extract(image):
 image=np.asarray(image)
 if image.shape!=(256,256) or image.dtype!=np.uint8:raise ValueError('Expected native256x256uint8gray')
 thresholds=np.quantile(image,QUANTILES,method='linear');values=[];fractions=[]
 for t in thresholds:
  mask=image<=t;fractions.append(float(mask.mean()))
  for phase in (mask,~mask):
   a=directional(phase);values.extend(np.column_stack((a.mean(0),a.std(0,ddof=0))).ravel())
 result=np.asarray(values,dtype=np.float64)
 assert result.shape==(120,) and np.isfinite(result).all() and ((result>=0)&(result<=1)).all()
 return result,{'thresholds':thresholds.tolist(),'mask_fractions':fractions}
