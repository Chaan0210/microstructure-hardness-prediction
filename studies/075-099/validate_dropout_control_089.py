"""Verify matched dropout-only fits, control replay, sample-level diagnostics."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dropout_control_089 as s
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
fv=s.fv;e=s.e

def main():
 plan=s.check();torch.set_num_threads(2);start=time.perf_counter();m=e.backbone.model();results={};models=0;cpuerr=0.;replays=[];all_sq=[];all_changes=[];commonids=None
 for split in s.old.s.SPLITS:
  d=s.old.inputs(split);out=s.OUT/split;r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:pp={n:z[n].copy() for n in z.files}
  assert pp['ids'].tolist()==d['ids'];np.testing.assert_array_equal(pp['y'],d['y']);np.testing.assert_array_equal(pp['fold'],d['fid']);np.testing.assert_array_equal(pp['reference009'],d['reference009']);np.testing.assert_array_equal(pp['reference011'],d['reference011']);np.testing.assert_array_equal(pp['control_head'],d['basehead'])
  if commonids is None:commonids=d['ids']
  assert commonids==d['ids']
  for k in range(5):
   tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert not set(d['g'][tr])&set(d['g'][va]);path=out/f'fold-{k}';rec=json.loads((path/'report.json').read_text());control=s.old.s.foldpath(split,k);cr=json.loads((control/'report.json').read_text());assert fv.sha(path/'report.json')==r['fold_report_sha256'][f'fold-{k}/report.json'];assert rec['config']==s.CFG and len(rec['curve'])==12;assert fv.sha(path/'model.pt')==rec['checkpoint_sha256'] and fv.sha(path/'predictions.npz')==rec['predictions_sha256'];a=torch.load(_repo_io(path/'model.pt'),weights_only=True,map_location='cpu');ca=torch.load(_repo_io(control/'model.pt'),weights_only=True,map_location='cpu');sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());assert a['config']==s.CFG and a['target_mean']==ym and a['target_std']==ys and a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va];assert a['seed']==s.SEED+k and a['base_seed']==s.SEED
   for n,val in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(a[n].numpy(),val);np.testing.assert_array_equal(ca[n].numpy(),val)
   orders=np.random.default_rng(s.SEED+k).permuted(np.tile(np.arange(len(tr)),(12,1)),axis=1);seeds=np.array([s.SEED+k+10000*(ep+1)+j for ep in range(12) for j in range(0,len(tr),2)],dtype=np.int64);assert rec['batch_order_sha256']==hashlib.sha256(orders.tobytes()).hexdigest() and rec['dropout_seeds_sha256']==hashlib.sha256(seeds.tobytes()).hexdigest()
   for n in ['initial_state_sha256','batch_order_sha256','dropout_seeds_sha256','fixed_state_sha256']:assert rec[n]==cr[n]
   model=e.Model(m,sc,s.SEED+k,False);model.head.net[2].p=.25;assert e.h.state_hash(model.state_dict())==rec['initial_state_sha256'] and e.fixed_hash(model)==rec['fixed_state_sha256'];assert model.head.net[2].p==s.CFG['dropout'] and e.CFG['dropout']==.5
   for n,v in model.state_dict().items():
    if not n.startswith('head.'):assert torch.equal(v,a['state'][n])
   assert e.h.state_hash(a['state'])==rec['final_state_sha256'];assert {n for n,p in model.named_parameters() if p.requires_grad}=={n for n,p in model.named_parameters() if n.startswith('head.')}
   for ep,c in enumerate(rec['curve']):assert c['head_lr']==cr['curve'][ep]['head_lr']==s.CFG['head_min_lr']+.5*(s.CFG['head_lr']-s.CFG['head_min_lr'])*(1+math.cos(math.pi*ep/12))
   model.load_state_dict(a['state']);model=model.to('mps');pred=e.predict(model,d,None,va)*ys+ym;tp=e.predict(model,d,None,tr)*ys+ym
   with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['train_indices'],tr);np.testing.assert_array_equal(z['validation_indices'],va);np.testing.assert_array_equal(z['predictions'],pred);np.testing.assert_array_equal(z['train_predictions'],tp)
   np.testing.assert_array_equal(pred,pp['new_head'][va]);assert fv.rmse(d['y'][tr],tp)==rec['curve'][-1]['train_RMSE'];model=model.cpu()
   with torch.no_grad():cp=model(torch.tensor(d['B'][va[:2]],dtype=torch.float32)).numpy().astype(float)*ys+ym
   ce=float(abs(cp-pred[:2]).max());assert ce<.003;cpuerr=max(cpuerr,ce);del model;torch.mps.empty_cache();models+=1
   if k==0:
    dd=dict(d);dd['y']=d['y'].copy();dd['y'][va]=10000+np.arange(len(va))
    for dropout,expected,expectedstate in [(.5,d['basehead'][va],cr['final_state_sha256']),(.25,pred,rec['final_state_sha256'])]:
     rp,rr=s.fit(dd,m,k,s.SEED,s.OUT/f'validation-replay-{split}-{dropout}',dropout=dropout);np.testing.assert_array_equal(rp,expected);assert rr['final_state_sha256']==expectedstate;replays.append({'split':split,'dropout':dropout,'outer_validation_labels_changed_only_in_memory':True,'full12epoch_state_exact':True,'prediction_error':0.})
  np.testing.assert_array_equal(pp['primary'],.9*d['reference009']+.1*pp['new_head']);comp,diag,advance,sq,ab,change=s.statistics(d,pp['primary']);assert comp==r['comparison'] and diag==r['sample_diagnostics'] and advance==r['advance'];np.testing.assert_array_equal(sq,pp['squared_error_gain']);np.testing.assert_array_equal(ab,pp['absolute_error_gain']);np.testing.assert_array_equal(change,pp['prediction_change']);assert r['scores']=={'reference011':fv.rmse(d['y'],d['reference011']),'primary':fv.rmse(d['y'],pp['primary']),'head':fv.rmse(d['y'],pp['new_head']),'control_head':fv.rmse(d['y'],d['basehead'])};results[split]={'scores':r['scores'],'comparison':comp,'sample_diagnostics':diag,'advance':advance,'pooled_train_RMSE':r['pooled_train_RMSE'],'report_sha256':fv.sha(out/'report.json'),'predictions_sha256':r['predictions_sha256']};all_sq.append(sq);all_changes.append(change)
 sq=np.array(all_sq);changes=np.array(all_changes);counts=(sq>0).sum(0);cross={'same400_rows_not_independent':True,'rows_improved_in_n_partitions':{str(k):int((counts==k).sum()) for k in range(4)},'rows_positive_mean_squared_error_gain':int((sq.mean(0)>0).sum()),'between_partition_perID_gain_correlations':np.corrcoef(sq).tolist(),'between_partition_prediction_change_correlations':np.corrcoef(changes).tolist()}
 with (s.OUT/'per-id-diagnostics.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(commonids),squared_error_gains_by_partition=sq,prediction_changes_by_partition=changes,improved_partition_count=counts)
 s.check();v={'passed':True,'eligible_for_candidate':all(r['advance'] for r in results.values()),'results':results,'models_verified':models,'max_MPS_reload_error':0.,'max_CPU_MPS_error':cpuerr,'replay':replays,'paired_initial_weights_orders_rng_seeds_exact':True,'dropout_only_source_comparison_passed':True,'foldtrain_only_scalers_targets_verified':True,'cross_partition_sample_diagnostics':cross,'per_id_diagnostics_sha256':fv.sha(s.OUT/'per-id-diagnostics.npz'),'protected011_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')),'limits':plan['limits'],'holdout_test_selection_access':False,'cost':0,'seconds':time.perf_counter()-start,'source_sha256':fv.sha(Path(__file__))};fv.write(_repo_path(ROOT, 'reports/dropout-control-089-validation.json'),v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
