"""Numerical fixtures and label-blind resource/contract checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import engine as e
import sys
import numpy as np
from fit_fusion import cohorts
def main():
 r=np.linspace(-3,3,40);R=np.c_[r,r,r,r,np.zeros(40)];Q=np.c_[np.linspace(12,20,40),np.linspace(3,7,40)];target=.3*r;rows=[]
 for lam in [1,10,100]:
  a=e.solve(R,Q,target,.4,'shared',lam);b=e.solve(R,Q,target,.4,'global',lam);error=float(np.max(abs(R[:,:1]@np.array(a['theta'])[:,0]-R@np.array(b['theta'])[:,0])));assert error<1e-5;rows.append({'lambda':lam,'duplicate_predictor_max_difference':error})
 for k in range(5):
  B,V,ix,ref,beta=cohorts(k);assert abs(e.incumbent_beta(B)-beta)<1e-12
  for kind in ['shared','global','conditional']:assert e.fit(B,kind,1,beta)['constraint_violation']<1e-7
 if not (c.HERE/'preflight.json').exists():c.save(_repo_io('preflight.json'),dict(passed=True,duplicate_control=rows,baseline_beta_all5_exact=True,actual_context_solver_all5_pass=True,query_accuracy_computed=False,code_sha256=c.sha(c.HERE/'engine.py')))
 else:
  import json
  assert json.loads((c.HERE/'preflight.json').read_text())['duplicate_control']==rows
 print('PREFLIGHT VERIFIED',rows)
if __name__=='__main__':sys.addaudithook(c.guard);main()
