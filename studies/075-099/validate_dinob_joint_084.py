"""Independent paired initialization, train-only scaling, reload and full joint replay audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_joint_084 as e
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
fv=e.fv

def main():
 plan=e.check();torch.set_num_threads(2);start=time.perf_counter();results={};models=0;reloaderr=0.;cpuerr=0.;replay=[];m=e.backbone.model()
 for split in ('original','20261031','20261101'):
  out=e.OUT/split
  if not (out/'report.json').exists():continue
  d=e.inputs(split);tokens=e.get_cache(d);r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:pp={n:z[n].copy() for n in z.files}
  assert pp['ids'].tolist()==d['ids'];np.testing.assert_array_equal(pp['y'],d['y']);np.testing.assert_array_equal(pp['fold'],d['fid'])
  for n in ('reference009','reference010'):np.testing.assert_array_equal(pp[n],d[n])
  for k in range(5):
   tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);fd=out/f'fold-{k}';pair=json.loads((fd/'report.json').read_text());assert fv.sha(fd/'report.json')==r['fold_report_sha256'][f'fold-{k}/report.json'];assert not set(d['g'][tr])&set(d['g'][va]);sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());orders=np.random.default_rng(e.CFG['seed']+k).permuted(np.tile(np.arange(len(tr)),(e.CFG['epochs'],1)),axis=1);seedarray=np.array([e.seed_for(k,ep,j) for ep in range(12) for j in range(0,len(tr),2)],dtype=np.int64);warm={}
   for arm in e.ARMS:
    path=fd/arm;rec=json.loads((path/'report.json').read_text());assert rec==pair['arms'][arm] and rec['config']==e.CFG and len(rec['curve'])==12;assert rec['batch_order_sha256']==hashlib.sha256(orders.tobytes()).hexdigest() and rec['dropout_seeds_sha256']==hashlib.sha256(seedarray.tobytes()).hexdigest();assert fv.sha(path/'model.pt')==rec['checkpoint_sha256'] and fv.sha(path/'predictions.npz')==rec['predictions_sha256'];a=torch.load(_repo_io(path/'model.pt'),weights_only=True,map_location='cpu');assert a['arm']==arm and a['config']==e.CFG and a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va] and a['target_mean']==ym and a['target_std']==ys
    for n,val in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(a[n].numpy(),val)
    model=e.Model(m,sc,e.CFG['seed']+k,arm=='joint');assert e.h.state_hash(model.state_dict())==rec['initial_state_sha256'];assert e.h.state_hash(model.head.state_dict())==rec['initial_head_sha256'];assert e.fixed_hash(model)==rec['fixed_state_sha256'];expected={n for n,p in model.named_parameters() if n.startswith('head.') or (arm=='joint' and n.startswith('block.'))};assert {n for n,p in model.named_parameters() if p.requires_grad}==expected
    for n,val in model.state_dict().items():
     if n not in expected:assert torch.equal(val,a['state'][n]),n
    assert rec['first_update_gradients']['head']>0
    if arm=='joint':assert rec['first_update_gradients']['block']>0 and rec['lastblock_delta_L2']>0
    else:assert rec['first_update_gradients']['block']==0 and rec['lastblock_delta_L2']==0
    for ep,c in enumerate(rec['curve']):
     for kind in ['head','block']:
      expectedlr=e.CFG[kind+'_min_lr']+.5*(e.CFG[kind+'_lr']-e.CFG[kind+'_min_lr'])*(1+math.cos(math.pi*ep/12)) if kind=='head' or arm=='joint' else 0.;assert c[kind+'_lr']==expectedlr
    assert e.h.state_hash(a['state'])==rec['final_state_sha256'];model=model.to('mps');warm[arm]=e.predict(model,d,tokens,tr[:8])*ys+ym;model.load_state_dict(a['state']);pred=e.predict(model,d,tokens,va)*ys+ym;trainpred=e.predict(model,d,tokens,tr)*ys+ym
    with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['train_indices'],tr);np.testing.assert_array_equal(z['validation_indices'],va);np.testing.assert_array_equal(z['warm_train8'],warm[arm]);np.testing.assert_array_equal(z['train_predictions'],trainpred);np.testing.assert_array_equal(z['predictions'],pred)
    er=float(abs(pred-pp[arm][va]).max());assert er<1e-5;reloaderr=max(reloaderr,er);assert rec['curve'][-1]['train_RMSE']==fv.rmse(d['y'][tr],trainpred);model=model.cpu();ii=va[:2]
    with torch.no_grad():cp=model(torch.tensor(d['B'][ii],dtype=torch.float32),torch.tensor(np.array(tokens[ii],copy=True)) if arm=='joint' else None).numpy().astype(float)*ys+ym
    ce=float(abs(cp-pred[:2]).max());assert ce<.003;cpuerr=max(cpuerr,ce);del model;torch.mps.empty_cache();models+=1
   np.testing.assert_allclose(warm['frozen'],warm['joint'],atol=1e-4,rtol=0)
   for n in ['initial_state_sha256','initial_head_sha256','batch_order_sha256','dropout_seeds_sha256']:assert pair['arms']['frozen'][n]==pair['arms']['joint'][n]
   if split=='original' and k==0:
    dd=dict(d);dd['y']=d['y'].copy();dd['y'][va]=10000+np.arange(len(va));new,rr=e.fit_pair(dd,tokens,m,k,e.OUT/'validation-replay-original');errors={}
    for arm in e.ARMS:
     errors[arm]=float(abs(new[arm]-pp[arm][va]).max());assert errors[arm]<1e-4;assert rr['arms'][arm]['final_state_sha256']==pair['arms'][arm]['final_state_sha256']
    replay.append({'outer_validation_targets_changed_only_in_memory':True,'full12epoch_both_arms_state_exact':True,'prediction_errors':errors})
  for arm in e.ARMS:np.testing.assert_array_equal(pp[arm+'_blend'],.9*d['reference009']+.1*pp[arm])
  pairs={'joint_vs_frozen':('frozen','joint'),'jointblend_vs_frozenblend':('frozen_blend','joint_blend'),'jointblend_vs009':('reference009','joint_blend'),'jointblend_vs010':('reference010','joint_blend')}
  for n,(a,b) in pairs.items():assert e.base.prior.old.comparison(d['y'],pp[a],pp[b],d['fid'])==r['comparisons'][n]
  for n,s in r['scores'].items():assert s=={'rmse':fv.rmse(d['y'],pp[n]),'fold_rmse':[fv.rmse(d['y'][d['fid']==k],pp[n][d['fid']==k]) for k in range(5)]}
  assert r['advance']==all(c['gain']>0 for c in r['comparisons'].values());results[split]={'scores':r['scores'],'comparisons':r['comparisons'],'advance':r['advance'],'report_sha256':fv.sha(out/'report.json'),'predictions_sha256':r['predictions_sha256']}
 selection=json.loads((e.OUT/'selection.json').read_text());assert selection['advance']==results['original']['advance'];e.check();r={'passed':True,'results':results,'paired_models_verified':models,'max_MPS_reload_error':reloaderr,'max_CPU_MPS_error':cpuerr,'identical_initial_states_orders_and_dropout_seed_streams':True,'foldtrain_only_scalers_targets_verified':True,'frozen_arm_backbone_unchanged':True,'joint_arm_only_head_and_lastblock_changed':True,'no_prediction_anchor':True,'fixed12epochs_no_outer_validation_selection':True,'fixed_blend_no_OOF_calibrator_fit':True,'leakage_replay':replay,'eligible_for_candidate':len(results)==3 and all(x['advance'] for x in results.values()),'protected009_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')),'protected010_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-010.csv')),'protected010_files_verified':len(plan['protected010']),'holdout_test_selection_access':False,'cost':0,'seconds':time.perf_counter()-start,'source_sha256':fv.sha(Path(__file__)),'limits':plan['bias']};fv.write(_repo_path(ROOT, 'reports/dinob-joint-084-validation.json'),r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
