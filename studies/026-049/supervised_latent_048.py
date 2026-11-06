"""Fixed4-component target-aware latent readout; all fitting fold-local."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.distance import pdist,cdist,squareform
from scipy.linalg import solve
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from joint_kernel_047 import CenteredKernel
class LatentKernel:
 def __init__(self,alpha=1.):self.alpha=alpha
 def fit(self,X,y):
  X=np.asarray(X,dtype=np.float64);y=np.asarray(y,dtype=np.float64);assert X.shape==(len(y),3456);self.scale=StandardScaler().fit(X);A=self.scale.transform(X);self.ymean=float(y.mean());self.ystd=float(y.std());assert self.ystd>0;z=(y-self.ymean)/self.ystd
  self.pls=PLSRegression(n_components=4,scale=False,max_iter=500,tol=1e-6).fit(A,z);self.Z=self.pls.transform(A);assert self.Z.shape==(len(y),4) and np.isfinite(self.Z).all();dist=pdist(self.Z,'sqeuclidean');self.gamma=1/float(np.median(dist[dist>1e-12]));raw=np.exp(-self.gamma*squareform(dist));self.center=CenteredKernel();K=self.center.fit(raw);self.kernel_train=K.copy();K.flat[::len(y)+1]+=self.alpha;self.coef=solve(K,z,assume_a='pos');self.n_train=len(y);return self
 def predict(self,X):
  Z=self.pls.transform(self.scale.transform(X));K=self.center.transform(np.exp(-self.gamma*cdist(Z,self.Z,'sqeuclidean')));return K@self.coef*self.ystd+self.ymean
 def linear_predict(self,X):return self.pls.predict(self.scale.transform(X)).reshape(-1)*self.ystd+self.ymean
