"""Algebraic checks only; synthetic arrays never train a competition model."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from fisher_pooling_067 import fisher_raw,normalize,sample_indices
from scipy.special import logsumexp

def reference(Z,w,mu,var):
 logp=-.5*np.sum(np.log(2*np.pi*var)[None]+(Z[:,None]-mu[None])**2/var[None],axis=2)+np.log(w)[None];q=np.exp(logp-logsumexp(logp,axis=1)[:,None]);n=len(Z);q0=q.sum(0);q1=q.T@Z;q2=q.T@(Z*Z);U=(q1-q0[:,None]*mu)/(n*np.sqrt(w)[:,None]*np.sqrt(var));V=(q2-2*mu*q1+q0[:,None]*(mu*mu-var))/(n*np.sqrt(2*w)[:,None]*var);return np.r_[U.ravel(),V.ravel()],q

def run():
 Z=np.array([[-1.,-2.],[1.,2.]]);raw=fisher_raw(Z,np.ones(1),np.zeros((1,2)),np.ones((1,2)),np.ones((2,1)));np.testing.assert_allclose(raw,[0,0,0,3/np.sqrt(2)],atol=1e-14,rtol=0);assert np.array_equal(normalize(np.zeros(4)),np.zeros(4))
 rng=np.random.default_rng(123);Z=rng.normal(size=(37,3));w=np.array([.3,.7]);mu=np.array([[-1.,0.,.5],[.7,-.3,1.]]);var=np.array([[.5,1.2,.8],[1.5,.7,2.]]);expected,q=reference(Z,w,mu,var);actual=fisher_raw(Z,w,mu,var,q);np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-13);order=rng.permutation(len(Z));np.testing.assert_allclose(fisher_raw(Z[order],w,mu,var,q[order]),actual,atol=1e-14,rtol=1e-13);np.testing.assert_allclose(fisher_raw(np.tile(Z,(2,1)),w,mu,var,np.tile(q,(2,1))),actual,atol=1e-14,rtol=1e-13);assert abs(np.linalg.norm(normalize(actual))-1)<1e-14
 for ident in ('TRAIN_000001','TRAIN_000099'):
  ix=sample_indices(ident);assert len(ix)==len(set(ix))==64 and ix.min()>=0 and ix.max()<1296;np.testing.assert_array_equal(ix,sample_indices(ident))
 return {'single_gaussian_analytic_fixture':True,'independent_sufficient_statistics_error':float(abs(expected-actual).max()),'token_permutation_and_duplication_invariant':True,'signedsqrt_L2_and_zero_fixture':True,'deterministic_equal64_sampling':True}
if __name__=='__main__':print(run())
