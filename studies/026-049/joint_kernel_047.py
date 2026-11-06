"""Fold-local additive/interaction kernels on frozen DINO and compact morphology."""
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
from sklearn.preprocessing import StandardScaler,MinMaxScaler
class CenteredKernel:
 def fit(self,K):
  self.column=K.mean(0);self.grand=float(K.mean());C=K-K.mean(1)[:,None]-self.column[None,:]+self.grand;self.scale=float(np.trace(C)/len(C));assert self.scale>1e-12;return C/self.scale
 def transform(self,K):return (K-K.mean(1)[:,None]-self.column[None,:]+self.grand)/self.scale
class JointKernel:
 def __init__(self,blocks,kind='additive',alpha=1.):self.blocks=blocks;self.kind=kind;self.alpha=alpha
 def fit(self,D,M,y):
  D=np.asarray(D,dtype=np.float64);M=np.asarray(M,dtype=np.float64);y=np.asarray(y,dtype=np.float64);assert D.shape==(len(y),3456) and M.shape==(len(y),26) and self.kind in ('additive','interaction') and self.alpha>0
  self.dscale=StandardScaler().fit(D);self.D=self.dscale.transform(D)
  self.mbound=MinMaxScaler(clip=True).fit(M);self.mscale=StandardScaler().fit(self.mbound.transform(M));b=np.array(self.blocks);assert len(b)==26;self.weights=np.array([1/np.sqrt(np.sum(b==a)) for a in b]);self.M=self.mscale.transform(self.mbound.transform(M))*self.weights
  dist=pdist(self.M,'sqeuclidean');positive=dist[dist>1e-12];assert len(positive)>0;self.gamma=1/float(np.median(positive));rawD=self.D@self.D.T/self.D.shape[1];rawM=np.exp(-self.gamma*squareform(dist))
  self.dc=CenteredKernel();self.mc=CenteredKernel();KD=self.dc.fit(rawD);KM=self.mc.fit(rawM)
  if self.kind=='interaction':self.ic=CenteredKernel();KI=self.ic.fit(KD*KM);K=.25*KD+.25*KM+.5*KI
  else:self.ic=None;K=.5*KD+.5*KM
  self.ymean=float(y.mean());self.ystd=float(y.std());assert self.ystd>0;self.n_train=len(y);self.kernel_train=K.copy();K.flat[::len(K)+1]+=self.alpha;self.coef=solve(K,(y-self.ymean)/self.ystd,assume_a='pos');assert np.isfinite(self.coef).all();return self
 def kernel(self,D,M):
  D=self.dscale.transform(np.asarray(D,dtype=np.float64));M=self.mscale.transform(self.mbound.transform(np.asarray(M,dtype=np.float64)))*self.weights
  KD=self.dc.transform(D@self.D.T/self.D.shape[1]);KM=self.mc.transform(np.exp(-self.gamma*cdist(M,self.M,'sqeuclidean')))
  return .5*KD+.5*KM if self.kind=='additive' else .25*KD+.25*KM+.5*self.ic.transform(KD*KM)
 def predict(self,D,M):return self.kernel(D,M)@self.coef*self.ystd+self.ymean
