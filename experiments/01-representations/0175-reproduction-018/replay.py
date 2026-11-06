"""Representative fresh same-seed neural/selection replay after label poisoning."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,os
from pathlib import Path
import numpy as np,torch
import train as t

def main():
 t.check();d=t.input_data();root=t.OUT/'original/fold-0/outer';r=t.read(root/'report.json');U=np.array(r['U']);Q=np.array(r['Q']);d['y'][Q]+=10000
 touched=t.filesystem_guard();p=t.node(d,U,Q,0,t.OUT/'replay-outer0-poison',True)
 with np.load(_repo_io(root/'predictions.npz')) as z:
  for key in ['P','short']:
   np.testing.assert_array_equal(z[key],p[key])
 states={}
 for file in sorted(root.glob('*.pt')):
  a=torch.load(_repo_io(file),map_location='cpu',weights_only=True);b=torch.load(_repo_io(t.OUT/'replay-outer0-poison'/file.name),map_location='cpu',weights_only=True);key='state_dict' if file.name.startswith('CNN') else ('state' if file.name=='short.pt' else 'model');states[file.name]=all(torch.equal(a[key][n],b[key][n]) for n in a[key]);assert states[file.name]
 out={'passed':True,'poisoned_outer80':True,'fresh_models':{'CNN':3,'long_MLP':6,'short_B':1,'KRR_final':1,'KRR_inner':27},'prediction_and_state_exact':True,'states':states,'scope_limit':'Representative outernode replay only; all other nodes have structural train/query/group audits and modelreload. Not a complete13node poison rerun.','file_accesses':sorted(touched)};t.write(t.OUT/'replay-validation.json',out);t.emit(out)
if __name__=='__main__':main()
