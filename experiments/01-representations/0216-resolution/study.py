from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();assert c.read(R/'preflight.json')['passed'];d=c.data();nd=c.data('native');old=P/'experiments/contrast-augmentation-206';ref=c.load(R/'frozen-reference.npz');pred={a:np.full(400,np.nan) for a in ['control','native']};records=[];folders={};reuse=True;reason=None
with c.Monitor('training',5400):
 p,r=e.fit(d,0,'control',R/'control-replay-fold0');prior=c.read(old/'fold-0-control/fit.json')
 try:
  for key in ['initial_state_SHA256','rng_start','rng_end','epoch_state_SHA256','averaged_state_SHA256']:assert r[key]==prior[key],key
  a=c.load(R/'control-replay-fold0/streams.npz');b=c.load(old/'fold-0-control/streams.npz')
  for key in ['order','rot','flip','lr','rng_CPU','rng_MPS']:np.testing.assert_array_equal(a[key],b[key])
  np.testing.assert_array_equal(a['values'][:,:2],b['values']);np.testing.assert_array_equal(p,c.load(old/'fold-0-control/predictions.npz')['validation_predictions'])
 except AssertionError as ex:reuse=False;reason=str(ex)
 if reuse:
  for k in range(5):
   T,V=c.indices(k);folder=old/f'fold-{k}-control';sd=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);assert sd['normalization']==c.norm(d,T);original=torch.load(_repo_io(P/f'artifacts/run-009/fold-{k}.pt'),map_location='cpu',weights_only=True);e.equal_tree(sd['state_dict'],original['state_dict']);m=e.initialize(k);m.load_state_dict(sd['state_dict']);vp,z=e.predict(m,d['images'],V,sd['normalization']);np.testing.assert_array_equal(vp,c.load(folder/'predictions.npz')['validation_predictions']);np.testing.assert_array_equal(vp,ref['old_first_CNN'][V]);pred['control'][V]=vp;folders[f'control-{k}']=str(folder.relative_to(P));del m
 else:
  T,V=c.indices(0);pred['control'][V]=p;folders['control-0']=str((R/'control-replay-fold0').relative_to(P))
  for k in range(1,5):
   T,V=c.indices(k);folder=R/f'fold-{k}-control';p,r=e.fit(d,k,'control',folder);pred['control'][V]=p;folders[f'control-{k}']=str(folder.relative_to(P))
 c.write(R/'control-reuse.json',{'reused':reuse,'reason':reason,'folders':folders});c.emit({'control_reused':reuse,'reason':reason})
 for k in range(5):
  T,V=c.indices(k);folder=R/f'fold-{k}-native';p,r=e.fit(nd,k,'native',folder);pred['native'][V]=p;records.append(r);folders[f'native-{k}']=str(folder.relative_to(P));prior=c.read(P/folders[f'control-{k}']/'fit.json')
  for key in ['initial_state_SHA256','rng_start','rng_end']:assert r[key]==prior[key],key
  a=c.load(folder/'streams.npz');b=c.load(P/folders[f'control-{k}']/'streams.npz')
  for key in ['order','rot','flip','lr','rng_CPU','rng_MPS']:np.testing.assert_array_equal(a[key],b[key])
 assert all(np.isfinite(p).all() for p in pred.values());c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],**pred);c.write(R/'training-complete.json',{'reused_controls':reuse,'folders':folders,'records':records,'scores_computed':False})
c.check()
