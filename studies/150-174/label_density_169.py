"""Train-only continuous Gaussian label density, a fixed mild weighting proposal."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
BANDWIDTH_STD=.4

def density(train_y,query_y,leave_one_out=False):
    t=np.asarray(train_y,float);q=np.asarray(query_y,float);bw=max(float(t.std())*BANDWIDTH_STD,1e-6)
    K=np.exp(-.5*((q[:,None]-t[None,:])/bw)**2)
    if leave_one_out:
        np.testing.assert_array_equal(t,q);np.fill_diagonal(K,0);den=len(t)-1
    else:den=len(t)
    return K.sum(1)/(den*bw*np.sqrt(2*np.pi))

def weights(train_y):
    rho=density(train_y,train_y,True);median=float(np.median(rho));safe=np.maximum(rho,max(median*.1,1e-30));raw=np.clip((median/safe)**.25,.8,1.25);w=raw/raw.mean()
    assert np.isfinite(w).all() and abs(w.mean()-1)<1e-12 and w.max()/w.min()<=1.5625+1e-12
    return w,{'bandwidth':max(float(np.std(train_y))*.4,1e-6),'density':rho.tolist(),'median_density':median,'raw_weights':raw.tolist(),'weight_min':float(w.min()),'weight_max':float(w.max()),'weight_ratio':float(w.max()/w.min()),'effective_sample_size':float(w.sum()**2/(w@w))}
