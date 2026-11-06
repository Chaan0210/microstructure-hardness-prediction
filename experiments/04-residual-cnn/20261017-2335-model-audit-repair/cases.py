"""Post-evaluation influential-source evidence; never used for model selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import sys,json
import numpy as np
def main():
 d=c.inputs();byid={str(i):j for j,i in enumerate(d['ids'])};keys=list(d['keys']);out={}
 for name,arm in [('retarget','retarget_matched'),('dense','pool_matched')]:
  z=np.load(_repo_io(c.HERE/(name+'-oof.npz')));base=z['old_matched'];pred=z[arm];y=z['y'];diff=(base-y)**2-(pred-y)**2;order=np.argsort(diff);table=[]
  for group,indices in [('largest_harms',order[:5]),('largest_helps',order[-5:][::-1])]:
   for j in indices:
    q=byid[str(z['ids'][j])];table.append(dict(group=group,ID=str(z['ids'][j]),fold=int(z['fold'][j]),hardness=float(y[j]),incumbent082=float(z['base082'][j]),matched_oldCNN=float(base[j]),candidate=float(pred[j]),SSE_gain_vs_matched=float(diff[j]),noise=float(d['noise'][q]),n_objects=float(d['X'][q,keys.index('n_obj')]),u_ph=float(d['X'][q,keys.index('u_ph')])))
  out[name]=table
 c.save(_repo_io('influential-cases.json'),dict(descriptive_only=True,comparison='Candidate minus matched oldCNN learning effect; both use the same076 calibration procedure',cases=out));print(json.dumps(out,indent=2))
if __name__=='__main__':sys.addaudithook(c.guard);main()
