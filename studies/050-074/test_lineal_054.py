"""Independent enumeration fixtures; artificial masks never train a regressor."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import lineal_features_054 as f

def brute_discrete(mask,dy,dx,k,endpoints_only=False):
 count=valid=0;ny,nx=mask.shape
 for y in range(ny):
  for x in range(nx):
   yy=y+dy*k;xx=x+dx*k
   if not (0<=yy<ny and 0<=xx<nx):continue
   valid+=1
   use=(0,k) if endpoints_only else range(k+1)
   count+=int(all(mask[y+dy*j,x+dx*j] for j in use))
 return count/valid if valid else 0.

def brute(mask,lengths):
 out=[]
 for dy,dx in f.DIRECTIONS:
  spacing=np.hypot(dy,dx);vals=[]
  for L in lengths:
   lag=L/spacing;lo=int(np.floor(lag));hi=int(np.ceil(lag));w=lag-lo
   vals.append((1-w)*brute_discrete(mask,dy,dx,lo)+w*brute_discrete(mask,dy,dx,hi))
  out.append(vals)
 return np.array(out)

def run():
 t=time.perf_counter();tests=[];lengths=(1.,2.,3.,4.);rng=np.random.default_rng(20261221);fixtures={'zeros':np.zeros((8,8),bool),'ones':np.ones((8,8),bool),'single_row':np.zeros((8,8),bool),'clipped_edge_runs':np.zeros((8,8),bool),'alternating_columns':np.indices((8,8))[1]%2==0,'random':rng.random((8,8))<.45}
 fixtures['single_row'][3,:]=True;fixtures['clipped_edge_runs'][0,:3]=True;fixtures['clipped_edge_runs'][0,-2:]=True
 errors=[]
 for name,mask in fixtures.items():
  got=f.directional(mask,lengths);want=brute(mask,lengths);err=float(abs(got-want).max());np.testing.assert_allclose(got,want,rtol=0,atol=1e-14);errors.append(err);tests.append(name+'_brute')
 np.testing.assert_array_equal(f.directional(fixtures['ones'],lengths),np.ones((4,4)));np.testing.assert_array_equal(f.directional(fixtures['zeros'],lengths),np.zeros((4,4)))
 one=f.directional(fixtures['single_row'],lengths);np.testing.assert_array_equal(one[0],np.full(4,1/8));expected=np.zeros((3,4));expected[1:,0]=(1-1/np.sqrt(2))/8;np.testing.assert_allclose(one[1:],expected,rtol=0,atol=1e-14);tests.append('analytic_single_row_including_sub_diagonal_lag_interpolation')
 edge=f.directional(fixtures['clipped_edge_runs'],(2.,));assert edge[0,0]==1/(8*6);tests.append('edge_truncated_runs_counted_without_wrap')
 m=fixtures['alternating_columns'];assert f.directional(m,(2.,))[0,0]==0 and brute_discrete(m,0,1,2,True)>0;tests.append('whole_segment_not_just_two_endpoints')
 expected=(1-(2/np.sqrt(2)-1))*brute_discrete(fixtures['random'],1,1,1)+(2/np.sqrt(2)-1)*brute_discrete(fixtures['random'],1,1,2);assert abs(f.directional(fixtures['random'],(2.,))[2,0]-expected)<1e-14;tests.append('sqrt2_physical_lag_interpolation')
 # Uniform input with tied quantiles: lower phase=all pixels; complement=empty.
 a,_=f.extract(np.full((256,256),127,np.uint8));q=a.reshape(5,2,6,2);np.testing.assert_array_equal(q[:,0,:,0],np.ones((5,6)));np.testing.assert_array_equal(q[:,1,:,0],np.zeros((5,6)));assert np.count_nonzero(q[:,:,:,1])==0;tests.append('uniform_quantile_ties_finite_exact')
 im=rng.integers(0,256,(256,256),dtype=np.uint8);original,_=f.extract(im);d4error=0.
 for k in range(4):
  for v in (np.rot90(im,k),np.fliplr(np.rot90(im,k))):
   value,_=f.extract(np.ascontiguousarray(v));d4error=max(d4error,float(abs(original-value).max()));np.testing.assert_allclose(original,value,rtol=0,atol=1e-14)
 tests.append('D4_summary_invariance')
 # Complement changes which phase occupies a segment; probabilities need not sum1.
 assert np.any(f.directional(m,lengths)+f.directional(~m,lengths)<1);tests.append('mixed_segments_not_counted_in_either_phase')
 assert len(f.NAMES)==len(set(f.NAMES))==120
 return {'passed':True,'tests':tests,'count':len(tests),'brute_max_error':max(errors),'D4_max_error':d4error,'seconds':time.perf_counter()-t,'no_synthetic_training':True}
if __name__=='__main__':print(json.dumps(run()),flush=True)
