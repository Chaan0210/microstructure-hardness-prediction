from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys
from pathlib import Path
import numpy as np
P=_RepoPath(__file__).resolve().parents[3];sys.path.insert(0,str(_repo_path(P, 'studies')))
from compact_regression import BalancedKernel
class PolicyKernel:
 def __init__(self,blocks,bandwidth,alpha):self.blocks=blocks;self.bandwidth=bandwidth;self.alpha=alpha
 def fit(self,X,y):
  # Only predeclared missing HP is NaN. All measured zero values remain zero.
  assert np.isfinite(X[:,1:]).all();good=np.isfinite(X[:,0]);self.impute_=float(np.median(X[good,0])) if good.any() else 0.;self.train_good_=int(good.sum());Z=self.transform(X);self.model_=BalancedKernel(self.blocks,self.bandwidth,self.alpha).fit(Z,y);return self
 def transform(self,X):
  z=X.copy();missing=~np.isfinite(z[:,0]);z[missing,0]=self.impute_;assert np.isfinite(z).all();return z
 def predict(self,X):return self.model_.predict(self.transform(X))
