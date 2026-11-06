from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();d=c.data();oof=c.load(R/'OOF.npz');checks=[]
with c.Monitor('validation',900):
 for k in range(5):
  T,V=c.indices(k)
  altered=dict(d);altered['y']=d['y'].copy();altered['y'][V]=np.nan;assert c.norm(altered,T)==c.norm(d,T)
  for arm in ['control','gamma']:
   folder=(P/'experiments/contrast-augmentation-206'/f'fold-{k}-control') if arm=='control' else R/f'fold-{k}-gamma';sd=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);pr=c.load(folder/'predictions.npz');st=c.load(folder/'streams.npz');norm=c.norm(d,T);assert sd['normalization']==norm and sd['train_ids']==d['ids'][T].tolist() and sd['validation_ids']==d['ids'][V].tolist();np.testing.assert_array_equal(oof[arm][V],pr['validation_predictions']);np.testing.assert_array_equal(np.bincount(st['order'].ravel(),minlength=320),np.full(320,80));assert st['order'].shape==(800,32)
   rng=np.random.default_rng(20261003+k);orders=[];rots=[];flips=[]
   for epoch in range(80):
    order=rng.permutation(320)
    for first in range(0,320,32):orders.append(order[first:first+32]);rots.append(int(rng.integers(4)));flips.append(bool(rng.random()<.5))
   np.testing.assert_array_equal(st['order'],orders);np.testing.assert_array_equal(st['rot'],rots);np.testing.assert_array_equal(st['flip'],flips);expectedLR=np.repeat(.0001+.5*(.002-.0001)*(1+np.cos(np.arange(80)*np.pi/80)),10);np.testing.assert_allclose(st['lr'],expectedLR,rtol=1e-12,atol=1e-15)
   avg=None
   for ep in range(76,81):
    state=torch.load(_repo_io(folder/f'epoch-{ep}.pt'),map_location='cpu',weights_only=True)
    if avg is None:avg={n:v/5 for n,v in state.items()}
    else:
     for n,v in state.items():avg[n]+=v/5
   assert all(torch.equal(v,sd['state_dict'][n]) for n,v in avg.items());maxcpu=0.;maxmps=0.
   for dev in ['cpu']+(['mps'] if c.device=='mps' else []):
    m=c.legacy.TinyCNN().to(dev);m.load_state_dict(sd['state_dict']);p,z=e.predict(m,d['images'],V,norm,dev);err=float(abs(p-pr['validation_predictions']).max());assert err<=(1e-10 if dev==c.device else .001)
    if dev=='cpu':maxcpu=err
    else:maxmps=err
    del m
   checks.append({'fold':k,'arm':arm,'normalization_train_only_exact':True,'gamma_strength_train_only':True,'checkpoint_average_exact':True,'sample_order_D4_LR_replayed':True,'CPU_prediction_max_error':maxcpu,'MPS_prediction_max_error':maxmps});c.emit({'reload_checked':folder.name,'CPU_max_error':maxcpu})
   if c.device=='mps':torch.mps.empty_cache()
 # Replay fold0 after poisoning all validation targets; neither preprocessing nor learning may observe them.
 T,V=c.indices(0);poison=[]
 for arm,value in [('gamma',np.nan)]:
  pd=dict(d);pd['y']=d['y'].copy();pd['y'][V]=value;folder=R/f'poison-fold0-{arm}';pred,r=e.fit(pd,0,arm,folder);old=c.read(R/f'fold-0-{arm}/fit.json');np.testing.assert_array_equal(pred,oof[arm][V]);assert r['averaged_state_SHA256']==old['averaged_state_SHA256'];a=c.load(folder/'streams.npz');b=c.load(R/f'fold-0-{arm}/streams.npz')
  for key in a:np.testing.assert_array_equal(a[key],b[key])
  poison.append({'arm':arm,'validation_targets':'NaN','full80epoch_state_predictions_and_streams_exact':True});c.emit({'poison_replay_exact':arm})
 c.write(R/'validation.json',{'passed':True,'checks':checks,'poison_replays':poison,'new_candidate_models':5,'reused_controls':5,'control_verification_fits':1,'label_poison_retrain_fits':1,'OOF_SHA256':c.sha(R/'OOF.npz'),'raw_data_access':c.access,'max_CPU_prediction_error':max(r['CPU_prediction_max_error'] for r in checks),'max_MPS_prediction_error':max(r['MPS_prediction_max_error'] for r in checks),'performance_scores_computed':False})
c.check()
