"""D4 closure and concrete predictor invariance for fixed151, development only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_view_mean_151 as q
np=q.np;e=q.e
q.torch.set_num_threads(2);q.check()
def transform(x,k,flip):
 z=np.rot90(x,k,axes=(2,3));return np.ascontiguousarray(z[:,:,:,::-1] if flip else z)
x=np.arange(25).reshape(1,1,5,5);orbit=[transform(x,k,f) for k in range(4) for f in (False,True)];keys={z.tobytes() for z in orbit};assert len(keys)==8
for z in orbit:assert {transform(z,k,f).tobytes() for k in range(4) for f in (False,True)}==keys
d=q.data('original');file=_repo_path(ROOT, 'artifacts/nested-calibration-074/original/fold-0/inner-0/cnn.pt');images=d['images'][:3];v=q.views.predict_views(images,file,'cpu');err=0.
for k in range(4):
 for f in (False,True):
  vv=q.views.predict_views(transform(images,k,f),file,'cpu');delta=float(abs(v.mean(0)-vv.mean(0)).max());assert delta<1e-4;err=max(err,delta)
r={'passed':True,'D4_group_closure_exact':True,'distinct_views':8,'CNN_mean_invariance_max_error':err,'scope':'Three development images only; single existing checkpoint. D4 of CNN inputs, not assertion of DINO equivariance or physical orientation invariance.','checkpoint_sha256':e.sha(file),'source_sha256':e.sha(Path(__file__)),'cost':0};e.write(q.OUT/'extra-validation.json',r);e.emit(r)
