"""Cache,matched parameters,fold-only moments,reload and leakage replay audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patch_replication_093 as s
import numpy as np,torch
fv=s.fv

def main():
 plan=s.check();torch.set_num_threads(2);start=time.perf_counter();d=s.inputs('original');tokens,mom=s.get_cache(d);momenterr=0.
 prior_v=json.loads((_repo_path(ROOT, 'reports/patch-attention-092-validation.json')).read_text());assert prior_v['passed'];results={};models=0;cpuerr=0.;replays=[];squared=[];changes=[]
 for split in s.SPLITS:
  out=s.OUT/split
  if not (out/'report.json').exists():continue
  d=s.inputs(split);r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:pp={n:z[n].copy() for n in z.files}
  assert pp['ids'].tolist()==d['ids'];np.testing.assert_array_equal(pp['y'],d['y']);np.testing.assert_array_equal(pp['fold'],d['fid']);np.testing.assert_array_equal(pp['reference011'],d['reference011'])
  for k in range(5):
   tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert not set(d['g'][tr])&set(d['g'][va]);fd=out/f'fold-{k}';pair=json.loads((fd/'report.json').read_text());assert fv.sha(fd/'report.json')==r['fold_report_sha256'][f'fold-{k}/report.json'];mean,scale,var=s.normalize(mom,tr);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());orders=np.random.default_rng(s.CFG['seed']+k).permuted(np.tile(np.arange(len(tr)),(20,1)),axis=1);seeds=np.array([s.seed_for(k,ep,j) for ep in range(20) for j in range(0,len(tr),2)],dtype=np.int64)
   for arm in s.ARMS:
    path=fd/arm;rec=json.loads((path/'report.json').read_text());assert rec==pair['arms'][arm] and rec['config']==s.CFG and len(rec['curve'])==20;assert fv.sha(path/'model.pt')==rec['checkpoint_sha256'] and fv.sha(path/'predictions.npz')==rec['predictions_sha256'];a=torch.load(_repo_io(path/'model.pt'),weights_only=True,map_location='cpu');assert a['arm']==arm and a['config']==s.CFG and a['seed']==s.CFG['seed']+k and a['target_mean']==ym and a['target_std']==ys and a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va]
    for n,v in [('scaler_mean',mean),('scaler_scale',scale),('scaler_var',var)]:np.testing.assert_array_equal(a[n].numpy(),v)
    assert rec['batch_order_sha256']==hashlib.sha256(orders.tobytes()).hexdigest() and rec['dropout_seeds_sha256']==hashlib.sha256(seeds.tobytes()).hexdigest();model=s.Pool(mean,scale,s.CFG['seed']+k,arm);assert sum(p.numel() for p in model.parameters())==rec['parameters']==12913;assert all(p.requires_grad for p in model.parameters());assert s.e.h.state_hash(model.state_dict())==rec['initial_state_sha256'];assert set(rec['first_parameter_gradient_norms'])==set(dict(model.named_parameters())) and all(v>0 for v in rec['first_parameter_gradient_norms'].values())
    for n in ['mean','scale','coords']:assert torch.equal(model.state_dict()[n],a['state'][n])
    assert s.e.h.state_hash(a['state'])==rec['final_state_sha256']
    for ep,c in enumerate(rec['curve']):assert c['lr']==s.CFG['min_lr']+.5*(s.CFG['lr']-s.CFG['min_lr'])*(1+math.cos(math.pi*ep/20))
    model.load_state_dict(a['state']);model=model.to('mps');pv,ess,mx=s.predict(model,tokens,va);pv=pv*ys+ym;tp,tess,tmx=s.predict(model,tokens,tr);tp=tp*ys+ym
    with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:
     for n,v in [('train_indices',tr),('validation_indices',va),('predictions',pv),('train_predictions',tp),('effective_patch_count',ess),('max_weight',mx),('train_effective_patch_count',tess),('train_max_weight',tmx)]:np.testing.assert_array_equal(z[n],v)
    np.testing.assert_array_equal(pv,pp[arm][va]);assert fv.rmse(d['y'][tr],tp)==rec['train_RMSE'];assert rec['effective_patch_count_quantiles']==np.quantile(ess,[0,.25,.5,.75,1]).tolist()
    if arm=='mean':np.testing.assert_allclose(ess,1296,atol=.01,rtol=0);np.testing.assert_allclose(mx,1/1296,atol=1e-9,rtol=0)
    model=model.cpu()
    with torch.no_grad():cp=model(torch.tensor(np.array(tokens[va[:2]],copy=True))).numpy().astype(float)*ys+ym
    ce=float(abs(cp-pv[:2]).max());assert ce<.003;cpuerr=max(cpuerr,ce);del model;torch.mps.empty_cache();models+=1
   for n in ['initial_state_sha256','batch_order_sha256','dropout_seeds_sha256','parameters']:assert pair['arms']['mean'][n]==pair['arms']['attention'][n]
   if k==0:
    dd=dict(d);dd['y']=d['y'].copy();dd['y'][va]=10000+np.arange(len(va));mm=mom.copy();mm[va]+=100;rp,rr=s.fit_pair(dd,tokens,mm,k,s.OUT/f'validation-replay-{split}');errors={}
    for arm in s.ARMS:np.testing.assert_array_equal(rp[arm],pp[arm][va]);assert rr['arms'][arm]['final_state_sha256']==pair['arms'][arm]['final_state_sha256'];errors[arm]=0.
    replays.append({'split':split,'outer_validation_labels_and_normalizer_moments_changed_only_in_memory':True,'both20epoch_states_exact':True,'prediction_errors':errors})
  for arm in s.ARMS:np.testing.assert_array_equal(pp[arm+'_blend'],.9*d['reference011']+.1*pp[arm])
  for n,(a,b) in {'attention_vs_mean':('mean','attention'),'attentionblend_vs_meanblend':('mean_blend','attention_blend'),'attentionblend_vs011':('reference011','attention_blend')}.items():assert r['comparisons'][n]==s.prior.e.base.prior.old.comparison(d['y'],pp[a],pp[b],d['fid'])
  for n,score in r['scores'].items():assert score=={'rmse':fv.rmse(d['y'],pp[n]),'fold_rmse':[fv.rmse(d['y'][d['fid']==k],pp[n][d['fid']==k]) for k in range(5)]}
  from dropout_control_089 import statistics
  cc,sd,material,sq,ab,change=statistics(d,pp['attention_blend']);assert cc==r['comparisons']['attentionblend_vs011'] and sd==r['sample_diagnostics'];advance=all(c['gain']>0 for c in r['comparisons'].values()) and cc['positive_folds']>=3;assert advance==r['advance'] and r['material_gate']==(advance and material);np.testing.assert_array_equal(sq,pp['squared_error_gain']);np.testing.assert_array_equal(ab,pp['absolute_error_gain']);np.testing.assert_array_equal(change,pp['prediction_change']);results[split]={'scores':r['scores'],'comparisons':r['comparisons'],'sample_diagnostics':sd,'advance':advance,'material_gate':r['material_gate'],'report_sha256':fv.sha(out/'report.json'),'predictions_sha256':r['predictions_sha256']};squared.append(sq);changes.append(change)
 s.check();sq=np.array(squared);count=(sq>0).sum(0);v={'passed':True,'eligible_for_candidate':False,'candidate_policy':'Reproducibility only; no promotion regardless of scores','results':results,'models_verified':models,'parameters_each':12913,'prior092_cache_audit_reused_by_hash':True,'prior092_validation_sha256':fv.sha(_repo_path(ROOT, 'reports/patch-attention-092-validation.json')),'max_MPS_reload_error':0.,'max_CPU_MPS_error':cpuerr,'identical_initial_states_orders_rng_seed_streams':True,'all_parameters_active_in_both_arms':True,'train_only_normalizers_targets':True,'position_coordinates_retained':True,'label_invariance_assumed':False,'replay':replays,'same_sample_consistency':{'partitions':len(results),'same400_not_independent':True,'improved_in_n_partitions':{str(k):int((count==k).sum()) for k in range(len(results)+1)}},'protected011_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')),'holdout_test_selection_access':False,'cost':0,'seconds':time.perf_counter()-start,'limits':plan['limits'],'source_sha256':fv.sha(Path(__file__))};fv.write(_repo_path(ROOT, 'reports/patch-replication-093-validation.json'),v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
