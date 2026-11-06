"""095 checkpoint/scaler/source validation and heldout-label perturbation replay."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,gc,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import train_early_adapter_095 as t
s=t.s;fv=t.fv;OUT=t.OUT;CFG=t.CFG
import numpy as np,torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler

def load_state(model,state):
 result=model.load_state_dict(state,strict=False);assert not result.unexpected_keys and all(n.startswith('backbone.') for n in result.missing_keys)

def replay(d,m,arm):
 dd=dict(d);dd['y']=d['y'].copy();dd['y'][d['fid']==0]+=12345;tr=np.flatnonzero(dd['fid']!=0);sc=StandardScaler().fit(dd['B'][tr]);ym=dd['y'][tr].mean();ys=dd['y'][tr].std();model=s.Model(m,sc,CFG['seed'],arm,False).to('mps');model.training_mode();opt=s.optimizer(model);s.schedule(opt,0);tokens=s.e.get_cache(dd) if arm=='late' else None;order=t.orders_for(0,320)[0]
 for first in (0,16):
  opt.zero_grad(set_to_none=True)
  for off,i in enumerate(tr[order[first:first+16]]):
   torch.manual_seed(s.seed_for(0,0,first+off));p=s.batch(model,dd,tokens,int(i));loss=F.smooth_l1_loss(p,s.tensor([(dd['y'][i]-ym)/ys]),beta=1.);(loss/16).backward()
  torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.)
  if arm!='frozen':torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.)
  opt.step()
 saved=torch.load(_repo_io(OUT/'original/fold-0'/arm/'first-two-updates.pt'),weights_only=True,map_location='cpu');now=s.checkpoint_state(model);assert saved.keys()==now.keys();err=max(float(abs(now[n]-saved[n]).max()) for n in now);assert err==0
 del model,opt,tokens;gc.collect();torch.mps.empty_cache();return {'arm':arm,'outer_labels_changed_only_in_memory':True,'effective_updates':2,'state_max_error':err,'scope':'First2update deterministicleakagecheck,not a full12epochreplay.'}

def validate(split):
 t.check();torch.set_num_threads(2);start=time.perf_counter();d=s.inputs(split);m=s.backbone.model().to('mps');frozenhash=s.h.state_hash(m.state_dict());r=json.loads((OUT/split/'report.json').read_text());assert fv.sha(OUT/split/'development_predictions.npz')==r['predictions_sha256'];maxreload=0.;maxcpu=0.;records=[];preds={a:np.full(400,np.nan) for a in s.ARMS}
 for k in range(5):
  tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);sc=StandardScaler().fit(d['B'][tr]);fr=json.loads((OUT/split/f'fold-{k}/report.json').read_text())
  for arm in s.ARMS:
   path=OUT/split/f'fold-{k}'/arm;rec=json.loads((path/'report.json').read_text());assert rec==fr['arms'][arm];assert fv.sha(path/'model.pt')==rec['checkpoint_sha256'] and fv.sha(path/'predictions.npz')==rec['predictions_sha256'];ck=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);assert ck['config']==CFG and ck['arm']==arm and ck['train_ids']==[d['ids'][i] for i in tr] and ck['validation_ids']==[d['ids'][i] for i in va];assert ck['target_mean']==float(d['y'][tr].mean()) and ck['target_std']==float(d['y'][tr].std());assert ck['backbone_sha256']==frozenhash
   for name,val in [('mean',sc.mean_),('scale',sc.scale_),('var',sc.var_)]:np.testing.assert_array_equal(ck['scaler_'+name].numpy(),val)
   model=s.Model(m,sc,CFG['seed']+k,arm,False).to('mps');assert s.h.state_hash(s.checkpoint_state(model))==rec['initial_state_sha256'];load_state(model,ck['state']);assert s.h.state_hash(s.checkpoint_state(model))==rec['final_state_sha256'];tokens=s.e.get_cache(d) if arm=='late' else None;pred=t.predict(model,d,tokens,va)*ck['target_std']+ck['target_mean']
   with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['train_indices'],tr);np.testing.assert_array_equal(z['validation_indices'],va);err=float(abs(pred-z['predictions']).max());assert err==0
   maxreload=max(maxreload,err);preds[arm][va]=pred
   # Independent raw CPU feature path spotcheck; no prefixcache for early.
   del tokens;model=model.to('cpu');model.eval();raw=torch.tensor(s.backbone.ds.inputs(d['ids'][va[0]],'full504'))
   with torch.no_grad():cpu=float(model.head((model.raw_features(raw)-model.mean)/model.scale)[0])*ck['target_std']+ck['target_mean']
   ce=abs(cpu-pred[0]);assert ce<.02;maxcpu=max(maxcpu,ce);records.append({'fold':k,'arm':arm,'MPS_reload_max_error':err,'CPU_raw_image_prediction_error':ce,'checkpoint_sha256':rec['checkpoint_sha256']});m=model.backbone.to('mps');del model,ck,raw;gc.collect();torch.mps.empty_cache()
  print(json.dumps({'validated_split':split,'fold':k,'max_reload':maxreload,'max_CPU_error':maxcpu}),flush=True)
 with np.load(_repo_io(OUT/split/'development_predictions.npz'),allow_pickle=False) as z:
  assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid'])
  for arm in s.ARMS:np.testing.assert_array_equal(z[arm],preds[arm]);np.testing.assert_array_equal(z[arm+'_blend'],.9*d['reference009']+.1*preds[arm])
  np.testing.assert_array_equal(z['reference011'],d['reference011'])
 replays=[replay(d,m,arm) for arm in s.ARMS] if split=='original' else []
 assert s.h.state_hash(m.state_dict())==frozenhash and all(p.grad is None for p in m.parameters());t.check();v={'passed':True,'split':split,'models_verified':len(records),'max_MPS_reload_error':maxreload,'max_CPU_raw_image_error':maxcpu,'models':records,'replay':replays,'backbone_weights_unchanged':True,'foldtrain_scalers_targets_ids_exact':True,'no_heldout_label_in_training':True,'source_sha256':fv.sha(Path(__file__)),'report_sha256':fv.sha(OUT/split/'report.json'),'predictions_sha256':r['predictions_sha256'],'seconds':time.perf_counter()-start,'holdout_test_selection_access':False,'cost':0};fv.write(_repo_path(ROOT, f'reports/early-adapter-095-{split}-validation.json'),v);print(json.dumps(v),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('split',choices=['original','20261031','20261101']);a=p.parse_args();validate(a.split)
