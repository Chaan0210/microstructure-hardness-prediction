"""Independent NumPy calculation of the masked-field objective."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import numpy as np,torch,sys
def main():
    rng=np.random.default_rng(118);p=rng.normal(size=(3,256,16)).astype(np.float32);t=rng.normal(size=p.shape).astype(np.float32);mask=e.mask_batch(3,rng);actual=e.losses(torch.from_numpy(p),torch.from_numpy(t),torch.from_numpy(mask),True).numpy();manual=[]
    for i in range(3):
        pp=p[i,mask[i]].astype(float).reshape(-1,4,4);tt=t[i,mask[i]].astype(float).reshape(-1,4,4);pm=pp.mean((1,2),keepdims=True);tm=tt.mean((1,2),keepdims=True);dc=float(((pm-tm)**2).mean());ac=float(((pp-pm-tt+tm)**2).mean());gx=float((np.diff(pp,axis=2)-np.diff(tt,axis=2)) .__pow__(2).mean());gy=float((np.diff(pp,axis=1)-np.diff(tt,axis=1)) .__pow__(2).mean());manual.append([dc,ac,(gx+gy)/2])
    error=float(np.max(abs(actual-np.array(manual))));assert error<2e-6;changed=t.copy();changed[~mask]+=100;other=e.losses(torch.from_numpy(p),torch.from_numpy(changed),torch.from_numpy(mask),True).numpy();assert np.array_equal(actual,other)
    e.save(_repo_io('loss-audit.json'),dict(all_passed=True,independent_NumPy_max_error=error,unmasked_target_change_has_no_loss_effect=True,encoder_source_sha256=e.sha(e.HERE/'encoder.py'),source_sha256=e.sha(__file__)))
    print('LOSS INDEPENDENTLY VERIFIED',error)
if __name__=='__main__':
    torch.set_num_threads(2);sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
