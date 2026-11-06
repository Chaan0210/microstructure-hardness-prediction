"""Fold-local PCA32/diagGMM8 mean+variance improved Fisher encoding."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import hashlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
DIM=32;K=8;PATCHES=64;SEED=20270107;ALPHA=512.

def sample_indices(ident):
 seed=int.from_bytes(hashlib.sha256(('fisher067:'+ident).encode()).digest()[:8],'little')
 return np.sort(np.random.default_rng(seed).choice(1296,PATCHES,replace=False))

def fisher_raw(Z,weights,means,variances,responsibilities):
 Z=np.asarray(Z,dtype=np.float64);w=np.asarray(weights,dtype=np.float64);mu=np.asarray(means,dtype=np.float64);var=np.asarray(variances,dtype=np.float64);q=np.asarray(responsibilities,dtype=np.float64)
 assert len(Z)>0 and mu.shape==var.shape==(len(w),Z.shape[1]) and q.shape==(len(Z),len(w));assert np.isfinite(Z).all() and np.isfinite(q).all() and (w>0).all() and (var>0).all();np.testing.assert_allclose(q.sum(1),1.,rtol=0,atol=1e-10)
 delta=(Z[:,None,:]-mu[None,:,:])/np.sqrt(var)[None,:,:]
 U=(q[:,:,None]*delta).sum(0)/(len(Z)*np.sqrt(w)[:,None]);V=(q[:,:,None]*(delta*delta-1)).sum(0)/(len(Z)*np.sqrt(2*w)[:,None]);out=np.concatenate([U.ravel(),V.ravel()]);assert np.isfinite(out).all();return out

def normalize(raw):
 a=np.sign(raw)*np.sqrt(np.abs(raw));norm=float(np.linalg.norm(a));return a/max(norm,1e-12)

class FisherDictionary:
 def fit(self,samples,train_ids):
  X=np.asarray(samples,dtype=np.float64);assert X.shape==(len(train_ids)*PATCHES,384) and len(set(train_ids))==len(train_ids);self.train_ids=list(train_ids);self.sample_sha256=hashlib.sha256(X.tobytes()).hexdigest();self.pca=PCA(n_components=DIM,whiten=False,svd_solver='randomized',iterated_power=4,random_state=SEED).fit(X);Z=self.pca.transform(X);self.gmm=GaussianMixture(n_components=K,covariance_type='diag',tol=1e-3,reg_covar=1e-4,max_iter=200,n_init=1,init_params='kmeans',random_state=SEED).fit(Z);assert self.gmm.converged_ and np.isfinite(self.gmm.lower_bound_) and (self.gmm.weights_>0).all() and (self.gmm.covariances_>0).all();return self
 def encode(self,tokens):
  P=np.asarray(tokens,dtype=np.float64);assert P.shape==(1296,384) and np.isfinite(P).all();Z=self.pca.transform(P);q=self.gmm.predict_proba(Z);out=normalize(fisher_raw(Z,self.gmm.weights_,self.gmm.means_,self.gmm.covariances_,q));assert out.shape==(512,) and np.isfinite(out).all();return out

class FisherHead:
 def fit(self,X,y):
  assert X.shape==(len(y),512);self.scaler=StandardScaler().fit(X);self.ym=float(y.mean());self.ys=float(y.std());assert self.ys>0;self.ridge=Ridge(alpha=ALPHA).fit(self.scaler.transform(X),(y-self.ym)/self.ys);return self
 def predict(self,X):return self.ridge.predict(self.scaler.transform(X))*self.ys+self.ym
