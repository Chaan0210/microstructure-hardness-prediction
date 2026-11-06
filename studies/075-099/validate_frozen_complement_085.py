"""Independent085 fold/scaling/reload and label-perturbation audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import frozen_complement_085 as s
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
fv=s.fv;e=s.e

def main():
 plan=s.check();torch.set_num_threads(2);start=time.perf_counter();m=e.backbone.model();results={};models=0;cpuerr=0.;reloaderr=0.;replays=[]
 for split in s.SPLITS:
  d=e.inputs(split);out=s.OUT/split;r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:pp={n:z[n].copy() for n in z.files}
  assert pp['ids'].tolist()==d['ids'];np.testing.assert_array_equal(pp['y'],d['y']);np.testing.assert_array_equal(pp['fold'],d['fid'])
  for name in ['reference009','reference010']:np.testing.assert_array_equal(pp[name],d[name])
  for k in range(5):
   tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert not set(d['g'][tr])&set(d['g'][va]);path=s.foldpath(split,k);rec=json.loads((path/'report.json').read_text());assert fv.sha(path/'report.json')==r['fold_report_sha256'][str(path.relative_to(ROOT)/'report.json')];assert rec['config']==e.CFG and rec['arm']=='frozen' and len(rec['curve'])==12;assert fv.sha(path/'model.pt')==rec['checkpoint_sha256'] and fv.sha(path/'predictions.npz')==rec['predictions_sha256'];a=torch.load(_repo_io(path/'model.pt'),weights_only=True,map_location='cpu');sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());assert a['config']==e.CFG and a['arm']=='frozen' and a['target_mean']==ym and a['target_std']==ys and a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va]
   for n,val in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(a[n].numpy(),val)
   orders=np.random.default_rng(e.CFG['seed']+k).permuted(np.tile(np.arange(len(tr)),(12,1)),axis=1);seeds=np.array([e.seed_for(k,ep,j) for ep in range(12) for j in range(0,len(tr),2)],dtype=np.int64);assert rec['batch_order_sha256']==hashlib.sha256(orders.tobytes()).hexdigest() and rec['dropout_seeds_sha256']==hashlib.sha256(seeds.tobytes()).hexdigest()
   model=e.Model(m,sc,e.CFG['seed']+k,False);assert e.h.state_hash(model.state_dict())==rec['initial_state_sha256'] and e.h.state_hash(model.head.state_dict())==rec['initial_head_sha256'] and e.fixed_hash(model)==rec['fixed_state_sha256'];assert {n for n,p in model.named_parameters() if p.requires_grad}=={n for n,p in model.named_parameters() if n.startswith('head.')}
   for n,v in model.state_dict().items():
    if not n.startswith('head.'):assert torch.equal(v,a['state'][n])
   assert rec['lastblock_delta_L2']==0 and rec['first_update_gradients']['block']==0 and rec['first_update_gradients']['head']>0 and e.h.state_hash(a['state'])==rec['final_state_sha256']
   for ep,c in enumerate(rec['curve']):assert c['block_lr']==0 and c['head_lr']==e.CFG['head_min_lr']+.5*(e.CFG['head_lr']-e.CFG['head_min_lr'])*(1+math.cos(math.pi*ep/12))
   model=model.to('mps');warm=e.predict(model,d,None,tr[:8])*ys+ym;model.load_state_dict(a['state']);pred=e.predict(model,d,None,va)*ys+ym;tp=e.predict(model,d,None,tr)*ys+ym
   with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['train_indices'],tr);np.testing.assert_array_equal(z['validation_indices'],va);np.testing.assert_array_equal(z['warm_train8'],warm);np.testing.assert_array_equal(z['predictions'],pred);np.testing.assert_array_equal(z['train_predictions'],tp)
   err=float(abs(pred-pp['frozen'][va]).max());assert err==0;reloaderr=max(reloaderr,err);assert fv.rmse(d['y'][tr],tp)==rec['curve'][-1]['train_RMSE'];model=model.cpu()
   with torch.no_grad():cp=model(torch.tensor(d['B'][va[:2]],dtype=torch.float32)).numpy().astype(float)*ys+ym
   ce=float(abs(cp-pred[:2]).max());assert ce<.003;cpuerr=max(cpuerr,ce);del model;torch.mps.empty_cache();models+=1
   if k==0:
    dd=dict(d);dd['y']=d['y'].copy();dd['y'][va]=10000+np.arange(len(va));rp,rr=e.fit_arm(dd,None,m,k,'frozen',s.OUT/f'validation-replay-{split}');er=float(abs(rp-pp['frozen'][va]).max());assert er==0 and rr['final_state_sha256']==rec['final_state_sha256'];replays.append({'split':split,'full12epochs_state_exact':True,'outer_validation_labels_perturbed_only_in_memory':True,'prediction_error':er})
  np.testing.assert_array_equal(pp['primary'],.9*pp['reference009']+.1*pp['frozen'])
  if split=='original':
   with np.load(_repo_io(e.OUT/'original/development_predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(pp['frozen'],z['frozen']);np.testing.assert_array_equal(pp['primary'],z['frozen_blend'])
  for n,score in r['scores'].items():assert score=={'rmse':fv.rmse(d['y'],pp[n]),'fold_rmse':[fv.rmse(d['y'][d['fid']==k],pp[n][d['fid']==k]) for k in range(5)]}
  for n,c in r['comparisons'].items():assert c==e.base.prior.old.comparison(d['y'],pp[n],pp['primary'],d['fid'])
  assert r['advance']==all(c['gain']>0 for c in r['comparisons'].values());results[split]={'scores':r['scores'],'comparisons':r['comparisons'],'advance':r['advance'],'report_sha256':fv.sha(out/'report.json'),'predictions_sha256':r['predictions_sha256']}
 s.check();r={'passed':True,'eligible_for_candidate':all(v['advance'] for v in results.values()),'results':results,'frozen_models_verified':models,'max_MPS_reload_error':reloaderr,'max_CPU_MPS_error':cpuerr,'leakage_replay':replays,'foldtrain_only_scaler_targets_verified':True,'backbone_unchanged':True,'original084_reproduced_exactly':True,'no_new_calibration':True,'fixed_blend':'.9*009+.1*frozen084','selection_bias':plan['selection_bias'],'limits':plan['limits'],'protected009_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')),'protected010_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-010.csv')),'holdout_test_selection_access':False,'cost':0,'seconds':time.perf_counter()-start,'source_sha256':fv.sha(Path(__file__))};fv.write(_repo_path(ROOT, 'reports/frozen-complement-085-validation.json'),r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
