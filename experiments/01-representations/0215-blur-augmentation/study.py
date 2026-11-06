from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();assert c.read(R/'preflight.json')['passed'];d=c.data();pred={a:np.full(400,np.nan) for a in ['control','blur']};ref=c.load(R/'frozen-reference.npz');records=[];reuse=[]
old=P/'experiments/contrast-augmentation-206'
with c.Monitor('training',1200):
 # Full fixed-budget control replay proves this code path before reusing the five controls.
 p,r=e.fit(d,0,'control',R/'control-replay-fold0');prior=c.read(old/'fold-0-control/fit.json')
 for key in ['initial_state_SHA256','rng_start','rng_end','epoch_state_SHA256','averaged_state_SHA256']:assert r[key]==prior[key],key
 a=c.load(R/'control-replay-fold0/streams.npz');b=c.load(old/'fold-0-control/streams.npz')
 for key in ['order','rot','flip','lr','rng_CPU','rng_MPS']:np.testing.assert_array_equal(a[key],b[key])
 np.testing.assert_array_equal(a['values'][:,:2],b['values']);np.testing.assert_array_equal(p,c.load(old/'fold-0-control/predictions.npz')['validation_predictions']);c.emit({'control80epoch_replay_exact':True})
 for k in range(5):
  T,V=c.indices(k);folder=old/f'fold-{k}-control';sd=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);pr=c.load(folder/'predictions.npz');assert sd['normalization']==c.norm(d,T)
  original=torch.load(_repo_io(P/f'artifacts/run-009/fold-{k}.pt'),map_location='cpu',weights_only=True);e.equal_tree(sd['state_dict'],original['state_dict'])
  m=e.initialize(k);m.load_state_dict(sd['state_dict']);vp,z=e.predict(m,d['images'],V,sd['normalization']);np.testing.assert_array_equal(vp,pr['validation_predictions']);np.testing.assert_array_equal(vp,ref['old_first_CNN'][V]);pred['control'][V]=vp;del m
  p,r=e.fit(d,k,'blur',R/f'fold-{k}-blur');pred['blur'][V]=p;records.append(r);prior=c.read(folder/'fit.json')
  for key in ['initial_state_SHA256','rng_start','rng_end']:assert r[key]==prior[key],key
  a=c.load(R/f'fold-{k}-blur/streams.npz');b=c.load(folder/'streams.npz')
  for key in ['order','rot','flip','lr','rng_CPU','rng_MPS']:np.testing.assert_array_equal(a[key],b[key])
  reuse.append({'fold':k,'control_path':str(folder.relative_to(P)),'model_SHA256':c.sha(folder/'model.pt'),'OOF_SHA256':c.sha(folder/'predictions.npz'),'same_initial_weights_and_all800_order_D4_LR_CPU_MPS_RNG':True,'old_run009_state_exact':True,'control_reload_OOF_exact':True})
 assert all(np.isfinite(p).all() for p in pred.values());c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],**pred);c.write(R/'training-complete.json',{'candidate_fits':5,'control_verification_fits':1,'reused_control_fits':5,'records':records,'control_reuse':reuse,'performance_scores_computed':False})
c.check()
