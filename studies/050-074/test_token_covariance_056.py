from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import token_covariance_056 as f

def run():
 t=time.perf_counter();rng=np.random.default_rng(20261226);R=f.projection();np.testing.assert_array_equal(R,f.projection());tests=['projection_seed_replay'];P=rng.normal(size=(90,384));a,C,D=f.describe(P,R);np.testing.assert_array_equal(C,C.T);assert np.linalg.eigvalsh(C).min()>-1e-12;tests.append('symmetric_PSD')
 centered=P-P.mean(0);want=R.T@(centered.T@centered/len(P))@R;err=float(abs(C-want).max());np.testing.assert_allclose(C,want,rtol=0,atol=1e-12);tests.append('projected_original_covariance_identity')
 np.testing.assert_allclose(C,np.cov(P@R,rowvar=False,bias=True),atol=1e-12,rtol=0);tests.append('independent_numpy_population_covariance')
 for value in (0.,17.):assert np.count_nonzero(f.describe(np.full((90,384),value),R)[0])==0
 tests.append('constant_zero_covariance')
 np.testing.assert_allclose(f.describe(P+1000.,R)[0],a,atol=1e-11,rtol=0);tests.append('centering_translation_invariance')
 np.testing.assert_allclose(f.describe(3*P,R)[0],9*a,atol=1e-11,rtol=0);tests.append('quadratic_scale_law')
 np.testing.assert_allclose(f.describe(P[rng.permutation(len(P))],R)[0],a,atol=1e-12,rtol=0);tests.append('token_permutation_invariance_position_loss')
 # Two channels have equal means/stds but opposite cross-correlation.
 A=np.zeros((4,384));A[:,0]=[-1,-1,1,1];A[:,1]=A[:,0];B=A.copy();B[:,1]*=-1
 np.testing.assert_array_equal(A.mean(0),B.mean(0));np.testing.assert_array_equal(A.std(0),B.std(0));av,_,ad=f.describe(A,R);bv,_,bd=f.describe(B,R);np.testing.assert_array_equal(ad,bd);assert np.linalg.norm(av-bv)>1e-3;tests.append('same_mean_std_different_crosschannel_statistics')
 assert len(f.NAMES)==len(set(f.NAMES))==528 and R.shape==(384,32);tests.append('schema528')
 return {'passed':True,'count':len(tests),'tests':tests,'projected_cov_identity_max_error':err,'seconds':time.perf_counter()-t,'synthetic_training':False}
if __name__=='__main__':print(json.dumps(run()))
