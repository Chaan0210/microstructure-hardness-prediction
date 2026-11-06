from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();assert c.read(R/'preflight.json')['passed'];old=P/'experiments/huber-repartition-212';out={};records=[]
with c.Monitor('training',7200):
 for split in c.SPLITS:
  d=c.data(split);nd=c.data(split,'native');root=R/split;ref=c.load(root/'frozen-reference.npz');pred={a:np.full(400,np.nan) for a in ['control','native']};folders={};reuse=True;reason=None;folder=root/'control-replay-fold0';p,r=e.fit(d,0,'control',folder);prior=c.read(old/split/'fold-0-control/fit.json')
  try:
   for key in ['initial_state_SHA256','rng_start','rng_end','epoch_state_SHA256','averaged_state_SHA256']:assert r[key]==prior[key],key
   aa=c.load(folder/'streams.npz');bb=c.load(old/split/'fold-0-control/streams.npz')
   for key in bb:np.testing.assert_array_equal(aa[key],bb[key])
   np.testing.assert_array_equal(p,ref['old_first_CNN'][c.indices(d,0)[1]]);e.equal_tree(torch.load(_repo_io(folder/'final-live-and-optimizer.pt'),map_location='cpu',weights_only=True),torch.load(_repo_io(old/split/'fold-0-control/final-live-and-optimizer.pt'),map_location='cpu',weights_only=True))
  except AssertionError as ex:reuse=False;reason=str(ex)
  if reuse:
   pred['control']=ref['old_first_CNN'].copy()
   for k in range(5):folders[f'control-{k}']=str((old/split/f'fold-{k}-control').relative_to(P))
  else:
   T,V=c.indices(d,0);pred['control'][V]=p;folders['control-0']=str(folder.relative_to(P))
   for k in range(1,5):
    T,V=c.indices(d,k);folder=root/f'fold-{k}-control';p,r=e.fit(d,k,'control',folder);pred['control'][V]=p;folders[f'control-{k}']=str(folder.relative_to(P))
  c.write(root/'control-reuse.json',{'reused':reuse,'reason':reason,'folders':folders});c.emit({'split':split,'controls_reused':reuse,'reason':reason})
  for k in range(5):
   T,V=c.indices(d,k);folder=root/f'fold-{k}-native';p,r=e.fit(nd,k,'native',folder);pred['native'][V]=p;records.append(r);folders[f'native-{k}']=str(folder.relative_to(P));prior=c.read(P/folders[f'control-{k}']/'fit.json')
   for key in ['initial_state_SHA256','rng_start','rng_end']:assert r[key]==prior[key],key
   a=c.load(folder/'streams.npz');b=c.load(P/folders[f'control-{k}']/'streams.npz')
   for key in ['order','rot','flip','lr','rng_CPU','rng_MPS']:np.testing.assert_array_equal(a[key],b[key])
  assert all(np.isfinite(p).all() for p in pred.values());c.arrays(root/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],**pred);out[split]={'reused_controls':reuse,'folders':folders,'OOF_SHA256':c.sha(root/'OOF.npz')}
 c.write(R/'training-complete.json',{'splits':out,'records':records,'scores_computed':False})
c.check()
