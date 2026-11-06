"""Independent flood-fill and geometry fixtures; no synthetic training."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import betti_features_055 as f

def components(mask,c):
 seen=set();out=[];ny,nx=mask.shape
 offsets=[(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if (dy or dx) and (c==2 or abs(dy)+abs(dx)==1)]
 for yy,xx in zip(*np.nonzero(mask)):
  if (yy,xx) in seen:continue
  stack=[(int(yy),int(xx))];seen.add((yy,xx));pixels=[]
  while stack:
   y,x=stack.pop();pixels.append((y,x))
   for dy,dx in offsets:
    a,b=y+dy,x+dx
    if 0<=a<ny and 0<=b<nx and mask[a,b] and (a,b) not in seen:seen.add((a,b));stack.append((a,b))
  out.append(pixels)
 return out

def independent(mask,c):
 ny,nx=mask.shape;holes=sum(all(y not in (0,ny-1) and x not in (0,nx-1) for y,x in pixels) for pixels in components(~mask,3-c))
 return len(components(mask,c)),holes

def run():
 start=time.perf_counter();tests=[];rng=np.random.default_rng(20261223)
 ring=np.zeros((11,11),bool);ring[2:9,2:9]=1;ring[4:7,4:7]=0
 opened=ring.copy();opened[:6,5]=0
 edge=np.ones((11,11),bool);edge[:6,5]=0
 fixtures={'empty':np.zeros((11,11),bool),'full':np.ones((11,11),bool),'ring':ring,'open_ring':opened,'edge_background':edge,'diagonal':np.eye(11,dtype=bool),'random':rng.random((11,11))<.5}
 for name,mask in fixtures.items():
  sd=f.signed_distance(mask)
  for d in f.DISTANCES:
   for c in (1,2):assert f.betti(sd>=d,c)==independent(sd>=d,c),(name,d,c)
  tests.append(name+'_all_excursions_floodfill')
 assert f.betti(ring,1)==(1,1) and f.betti(opened,1)==(1,0) and f.betti(edge,1)==(1,0);tests.append('holes_exclude_boundary_background')
 assert f.betti(fixtures['diagonal'],1)==(11,0) and f.betti(fixtures['diagonal'],2)==(1,0);tests.append('four_eight_connectivity_duality')
 # Verify EDT values against in-image opposite-pixel distances independently.
 m=fixtures['random'];sd=f.signed_distance(m)
 for y,x in np.ndindex(m.shape):
  other=np.argwhere(m!=m[y,x]);v=np.sqrt(((other-[y,x])**2).sum(1)).min()*(1 if m[y,x] else -1);assert sd[y,x]==v
 tests.append('signed_EDT_in_image_opposite_pixels')
 for name in ('empty','full'):
  a=f.curves(fixtures[name]).reshape(11,2,2);assert np.isfinite(a).all();assert np.all(a[:,:,1]==0);assert np.all(a[:,:,0]==(1/121 if name=='full' else 0))
 tests.append('uniform_phase_infinite_distance_finite_curves')
 # Same raw component/hole counts; thin bridge disappears under erosion.
 thin=np.zeros((21,21),bool);thin[5:16,2:9]=1;thin[5:16,12:19]=1;thin[10,9:12]=1
 thick=thin.copy();thick[7:14,9:12]=1
 assert f.betti(thin,1)==f.betti(thick,1)==(1,0);assert f.betti(f.signed_distance(thin)>=2,1)[0]==2 and f.betti(f.signed_distance(thick)>=2,1)[0]==1
 tests.append('curves_distinguish_equal_zero_distance_counts')
 im=rng.integers(0,256,(256,256),dtype=np.uint8);a,_=f.extract(im)
 for k in range(4):
  for v in (np.rot90(im,k),np.fliplr(np.rot90(im,k))):np.testing.assert_array_equal(a,f.extract(np.ascontiguousarray(v))[0])
 tests.append('D4_invariance')
 a,_=f.extract(np.full((256,256),127,np.uint8));assert np.isfinite(a).all();assert len(f.NAMES)==len(set(f.NAMES))==132;assert len(f.ZERO_COLUMNS)==12;tests.append('quantile_ties_and_feature_layout')
 return {'passed':True,'count':len(tests),'tests':tests,'seconds':time.perf_counter()-start,'no_synthetic_training':True}
if __name__=='__main__':print(json.dumps(run()))
