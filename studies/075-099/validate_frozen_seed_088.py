"""Seed-only change,45head reload and perpartition leakage/replay validation."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import frozen_seed_088 as s
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
fv=s.fv;e=s.e

def main():
 plan=s.check();torch.set_num_threads(2);start=time.perf_counter();m=e.backbone.model();results={};models=0;cpuerr=0.;replays=[]
 for split in s.s.SPLITS:
  d=s.inputs(split);out=s.OUT/split;r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:pp={n:z[n].copy() for n in z.files}
  assert pp['ids'].tolist()==d['ids'];np.testing.assert_array_equal(pp['y'],d['y']);np.testing.assert_array_equal(pp['fold'],d['fid']);np.testing.assert_array_equal(pp['reference009'],d['reference009']);np.testing.assert_array_equal(pp['reference011'],d['reference011'])
  for si,seed in enumerate(s.SEEDS):
   for k in range(5):
    tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert not set(d['g'][tr])&set(d['g'][va]);path=s.path_for(split,seed,k);rec=json.loads((path/'report.json').read_text());assert fv.sha(path/'report.json')==r['fold_report_sha256'][str((path/'report.json').relative_to(ROOT))];assert rec['config']==s.CFG and len(rec['curve'])==12;assert fv.sha(path/'model.pt')==rec['checkpoint_sha256'] and fv.sha(path/'predictions.npz')==rec['predictions_sha256'];a=torch.load(_repo_io(path/'model.pt'),weights_only=True,map_location='cpu');sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());assert a['config']==s.CFG and a['target_mean']==ym and a['target_std']==ys and a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va]
    if si:assert a['seed']==rec['seed']==seed+k and a['base_seed']==seed
    for n,val in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(a[n].numpy(),val)
    orders=np.random.default_rng(seed+k).permuted(np.tile(np.arange(len(tr)),(12,1)),axis=1);seeds=np.array([seed+k+10000*(ep+1)+j for ep in range(12) for j in range(0,len(tr),2)],dtype=np.int64);assert rec['batch_order_sha256']==hashlib.sha256(orders.tobytes()).hexdigest() and rec['dropout_seeds_sha256']==hashlib.sha256(seeds.tobytes()).hexdigest();model=e.Model(m,sc,seed+k,False);assert e.h.state_hash(model.state_dict())==rec['initial_state_sha256'] and e.fixed_hash(model)==rec['fixed_state_sha256'];assert {n for n,p in model.named_parameters() if p.requires_grad}=={n for n,p in model.named_parameters() if n.startswith('head.')}
    for n,v in model.state_dict().items():
     if not n.startswith('head.'):assert torch.equal(v,a['state'][n])
    assert e.h.state_hash(a['state'])==rec['final_state_sha256']
    for ep,c in enumerate(rec['curve']):assert c['head_lr']==s.CFG['head_min_lr']+.5*(s.CFG['head_lr']-s.CFG['head_min_lr'])*(1+math.cos(math.pi*ep/12))
    model.load_state_dict(a['state']);model=model.to('mps');pred=e.predict(model,d,None,va)*ys+ym;tp=e.predict(model,d,None,tr)*ys+ym
    with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['train_indices'],tr);np.testing.assert_array_equal(z['validation_indices'],va);np.testing.assert_array_equal(z['predictions'],pred);np.testing.assert_array_equal(z['train_predictions'],tp)
    np.testing.assert_array_equal(pred,pp['seed_predictions'][si,va]);assert fv.rmse(d['y'][tr],tp)==rec['curve'][-1]['train_RMSE'];model=model.cpu()
    with torch.no_grad():cp=model(torch.tensor(d['B'][va[:2]],dtype=torch.float32)).numpy().astype(float)*ys+ym
    ce=float(abs(cp-pred[:2]).max());assert ce<.003;cpuerr=max(cpuerr,ce);del model;torch.mps.empty_cache();models+=1
    if k==0 and (si==1 or (split=='original' and si==0)):
     dd=dict(d);dd['y']=d['y'].copy();dd['y'][va]=10000+np.arange(len(va));rp,rr=s.fit(dd,m,k,seed,s.OUT/f'validation-replay-{split}-{seed}');np.testing.assert_array_equal(rp,pred);assert rr['final_state_sha256']==rec['final_state_sha256'];replays.append({'split':split,'seed':seed,'outer_validation_labels_changed_only_in_memory':True,'full12epoch_state_exact':True,'prediction_error':0.})
  np.testing.assert_array_equal(pp['seed_predictions'][0],d['basehead']);np.testing.assert_array_equal(pp['mean3'],pp['seed_predictions'].mean(0));np.testing.assert_array_equal(pp['primary'],.9*d['reference009']+.1*pp['mean3']);comp=e.base.prior.old.comparison(d['y'],d['reference011'],pp['primary'],d['fid']);assert comp==r['comparison'];assert r['scores']=={n:fv.rmse(d['y'],pp[n]) for n in ('reference011','primary')}|{'head_mean3':fv.rmse(d['y'],pp['mean3'])};assert r['advance']==(comp['gain']>0 and comp['positive_folds']>=3);var=float(np.mean(np.var(pp['seed_predictions'],axis=0)));assert var==r['seed_prediction_variance_mean'];assert abs(r['head_average_individual_MSE']-r['head_mean3_MSE']-var)<1e-10;results[split]={'scores':r['scores'],'comparison':comp,'advance':r['advance'],'report_sha256':fv.sha(out/'report.json'),'predictions_sha256':r['predictions_sha256'],'seed_prediction_variance_mean':var,'individual_head_scores_diagnostic_only':r['individual_head_scores_diagnostic_only']}
 s.check();v={'passed':True,'eligible_for_candidate':all(r['advance'] for r in results.values()),'results':results,'models_verified':models,'max_MPS_reload_error':0.,'max_CPU_MPS_error':cpuerr,'replay':replays,'original_base_seed_replay_exact':True,'foldtrain_only_scalers_targets_verified':True,'only_seed_changed':True,'no_seed_selection':True,'seeds':list(s.SEEDS),'config':s.CFG,'protected011_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')),'protected011_files':len(plan['protected011']),'selection_bias':plan['selection_bias'],'holdout_test_selection_access':False,'cost':0,'seconds':time.perf_counter()-start,'source_sha256':fv.sha(Path(__file__))};fv.write(_repo_path(ROOT, 'reports/frozen-seed-088-validation.json'),v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
