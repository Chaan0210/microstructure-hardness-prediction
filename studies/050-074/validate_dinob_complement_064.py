"""Validate sole equal-average recipe and every newly fitted B checkpoint."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_complement_064 as e
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
fv=e.fv

def main():
 t=time.perf_counter();plan=e.check();torch.set_num_threads(2);results={};count=0;replays=[];max_error=0.
 for split in ('original','20261031','20261101'):
  out=e.OUT/split
  if not out.exists():continue
  ids,S,B,y,g,fid,r003,r007,Smean=e.old.inputs(split);r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  assert d['ids'].tolist()==ids;np.testing.assert_array_equal(d['y'],y);np.testing.assert_array_equal(d['fold'],fid);np.testing.assert_array_equal(d['Smean'],Smean);np.testing.assert_array_equal(d['reference007'],r007);np.testing.assert_array_equal(d['reference003'],r003);source=_repo_path(ROOT, r['B_model_source']);train_errors=[]
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);sc=StandardScaler().fit(B[tr]);ym=float(y[tr].mean());ys=float(y[tr].std());edges=np.quantile(y[tr],np.arange(1,7)/7);log=json.loads((source/f'fold-{k}/report.json').read_text());oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}/report.json')).read_text());tp=[]
   for si,seed in enumerate(e.SEEDS):
    path=source/f'fold-{k}/baseB-seed-{seed}.pt';rec=log['models'][f'baseB-{seed}'];assert fv.sha(path)==rec['checkpoint_sha256'];a=torch.load(_repo_io(path),weights_only=True,map_location='cpu');assert a['input_dim']==6912 and a['seed']==seed+k and a['config']==e.h.CONFIG and a['effective_aux_weight']==0.;assert a['train_ids']==[ids[j] for j in tr] and a['validation_ids']==[ids[j] for j in va]
    for key,actual in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_),('bin_edges_unused',edges)]:np.testing.assert_array_equal(a[key].numpy(),actual)
    assert a['target_mean']==ym and a['target_std']==ys;assert e.h.state_hash(a['model'])==rec['final_state_sha256'];assert rec['batch_order_sha256']==oldlog['models'][str(seed)]['batch_order_sha256'];assert rec['classifier_max_parameter_change']==0
    m=e.h.Head(6912);m.load_state_dict(a['model']);p=e.h.predict(m,sc.transform(B[va]),ym,ys);np.testing.assert_array_equal(p,d['Bseeds'][si,va]);max_error=max(max_error,float(abs(p-d['Bseeds'][si,va]).max()));assert fv.rmse(y[va],p)==rec['validation_rmse'];tp.append(e.h.predict(m,sc.transform(B[tr]),ym,ys));count+=1
    if k==0 and si==0:
     _,_,_,xt,yt,bt,_=e.old.prep(B[tr],y[tr]);m2,r2=e.h.fit(xt,yt,bt,seed+k,0.);assert r2=={key:value for key,value in rec.items() if key not in ('checkpoint_sha256','validation_rmse')};np.testing.assert_array_equal(e.h.predict(m2,sc.transform(B[va]),ym,ys),p)
     # An unchanged S trajectory independently anchors each partition's historical control.
     _,sym,sys_,sxt,syt,sbt,_=e.old.prep(S[tr],y[tr]);sm,sr=e.h.fit(sxt,syt,sbt,seed+k,0.);old=oldlog['models'][str(seed)]
     for key in ('initial_state_sha256','batch_order_sha256','final_state_sha256'):assert sr[key]==old[key]
     replays.append({'split':split,'B100epoch_exact':True,'S100epoch_historical_state_exact':True,'max_prediction_error':0.})
   train_errors.extend((np.mean(tp,axis=0)-y[tr]).tolist())
  np.testing.assert_array_equal(d['Bmean'],d['Bseeds'].mean(0));np.testing.assert_array_equal(d['Bprimary'],.75*r003+.25*d['Bmean']);np.testing.assert_array_equal(d['mixed50'],.5*r007+.5*d['Bprimary']);np.testing.assert_allclose(d['mixed50'],.75*r003+.125*Smean+.125*d['Bmean'],atol=1e-12,rtol=0)
  for key,score in r['scores'].items():assert score=={'rmse':fv.rmse(y,d[key]),'fold_rmse':[fv.rmse(y[fid==k],d[key][fid==k]) for k in range(5)]}
  p=d['mixed50'];fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],p[fid==k]) for k in range(5)];ix=np.random.default_rng(20270104).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-p)[ix]**2,1));comp={'gain':fv.rmse(y,r007)-fv.rmse(y,p),'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist()};assert comp==r['comparison'];corr=float(np.corrcoef(y-r007,y-d['Bprimary'])[0,1]);assert corr==r['complementarity']['prediction_residual_correlation'];actual=float(np.mean((y-p)**2));avg=float(np.mean(((y-r007)**2+(y-d['Bprimary'])**2)/2));diversity=float(np.mean((r007-d['Bprimary'])**2)/4);np.testing.assert_allclose(actual,avg-diversity,atol=1e-10,rtol=0);assert r['complementarity']['mixture_MSE']==actual and r['complementarity']['disagreement_MSE_reduction']==diversity
  if split=='original':
   oldreport=json.loads((e.old.OUT/'original/report.json').read_text());ob=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-d['Bprimary'])[ix]**2,1));assert float(ob.std())==oldreport['comparison']['bootstrap_sd']==plan['previous_gate_explanation']['value']
  results[split]={'scores':r['scores'],'comparison':comp,'complementarity':r['complementarity'],'train_Bmean_rmse_pooled':float(np.sqrt(np.mean(np.array(train_errors)**2))),'report_sha256':fv.sha(out/'report.json'),'predictions_sha256':r['predictions_sha256']}
 sel=json.loads((e.OUT/'selection.json').read_text());assert sel['advance']==(results['original']['comparison']['gain']>0)
 if not sel['advance']:assert set(results)=={'original'}
 confirmed=len(results)==3 and all(v['comparison']['gain']>0 and v['comparison']['positive_folds']>=3 for v in results.values());e.check();v={'passed':True,'results':results,'models_verified':count,'max_reload_error':max_error,'full_replays':replays,'confirmed_for_final_fit':confirmed,'sole_recipe':{'candidate007':.5,'ViTB_primary063':.5},'plan_sha256':fv.sha(e.OUT/'plan.json'),'validation_source_sha256':fv.sha(Path(__file__)),'seconds':time.perf_counter()-t,'limits':fv.CAVEAT,'holdout_test_selection_access':False};fv.write(_repo_path(ROOT, 'reports/dinob-complement-064-validation.json'),v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
