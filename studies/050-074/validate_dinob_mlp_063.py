"""Reload every063 model, rederive fold-only scalers, replay fixed training."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_mlp_063 as e
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
fv=e.fv

def main():
 t=time.perf_counter();plan=e.check();torch.set_num_threads(2);count=0;results={};replays=[]
 for split in ('original','20261031','20261101'):
  out=e.OUT/split
  if not out.exists():continue
  ids,S,B,y,g,fid,r003,r007,oldmean=e.inputs(split);report=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==report['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  assert d['ids'].tolist()==ids;np.testing.assert_array_equal(d['y'],y);np.testing.assert_array_equal(d['fold'],fid);trainerrors={a:[] for a in ('controlS','baseB')};maxerr=0.
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);fd=out/f'fold-{k}';log=json.loads((fd/'report.json').read_text());oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}/report.json')).read_text())
   for arm,X in [('controlS',S),('baseB',B)]:
    sc=StandardScaler().fit(X[tr]);ym=float(y[tr].mean());ys=float(y[tr].std());edges=np.quantile(y[tr],np.arange(1,7)/7,method='linear');tp=[]
    for si,seed in enumerate(e.SEEDS):
     path=fd/f'{arm}-seed-{seed}.pt';r=log['models'][f'{arm}-{seed}'];old=oldlog['models'][str(seed)];assert fv.sha(path)==r['checkpoint_sha256'];a=torch.load(_repo_io(path),weights_only=True,map_location='cpu');assert a['train_ids']==[ids[j] for j in tr] and a['validation_ids']==[ids[j] for j in va];assert a['seed']==seed+k and a['input_dim']==X.shape[1] and a['config']==e.h.CONFIG and a['effective_aux_weight']==0.
     for key,actual in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_),('bin_edges_unused',edges)]:np.testing.assert_array_equal(a[key].numpy(),actual)
     assert a['target_mean']==ym and a['target_std']==ys;assert e.h.state_hash(a['model'])==r['final_state_sha256'];assert r['batch_order_sha256']==old['batch_order_sha256'] and r['classifier_max_parameter_change']==0
     if arm=='controlS':
      for key in ('initial_state_sha256','final_state_sha256'):assert r[key]==old[key]
     m=e.h.Head(X.shape[1]);m.load_state_dict(a['model']);pv=e.h.predict(m,sc.transform(X[va]),ym,ys);np.testing.assert_array_equal(pv,d[arm+'_seeds'][si,va]);maxerr=max(maxerr,float(abs(pv-d[arm+'_seeds'][si,va]).max()));tp.append(e.h.predict(m,sc.transform(X[tr]),ym,ys));assert r['validation_rmse']==fv.rmse(y[va],pv);count+=1
     if k==0 and si==0:
      _,_,_,xt,yt,bt,_=e.prep(X[tr],y[tr]);m2,r2=e.h.fit(xt,yt,bt,seed+k,0.);assert r2=={key:value for key,value in r.items() if key not in ('checkpoint_sha256','validation_rmse')};np.testing.assert_array_equal(e.h.predict(m2,sc.transform(X[va]),ym,ys),pv);replays.append({'split':split,'arm':arm,'seed':seed,'parameter_hash_exact':True,'prediction_error':0.})
    trainerrors[arm].extend((np.mean(tp,axis=0)-y[tr]).tolist())
  for arm in ('controlS','baseB'):np.testing.assert_array_equal(d[arm],d[arm+'_seeds'].mean(0));assert np.isfinite(d[arm+'_seeds']).all();assert report['train_rmse_pooled'][arm]==float(np.sqrt(np.mean(np.array(trainerrors[arm])**2)))
  np.testing.assert_array_equal(d['primary'],.75*r003+.25*d['baseB']);np.testing.assert_array_equal(d['control007'],.75*r003+.25*d['controlS']);np.testing.assert_allclose(d['control007'],r007,atol=1e-8,rtol=0);np.testing.assert_array_equal(d['reference007'],r007);np.testing.assert_array_equal(d['reference003'],r003);np.testing.assert_allclose(d['controlS'],oldmean,atol=1e-8,rtol=0)
  for key,score in report['scores'].items():assert score=={'rmse':fv.rmse(y,d[key]),'fold_rmse':[fv.rmse(y[fid==k],d[key][fid==k]) for k in range(5)]}
  # Recompute comparisons independently from the authoring helper.
  comps={}
  for name in ('primary','equal_SB'):
   if name not in d:continue
   if name=='equal_SB':np.testing.assert_array_equal(d[name],.75*r003+.125*d['controlS']+.125*d['baseB'])
   p=d[name];fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],p[fid==k]) for k in range(5)];ix=np.random.default_rng(20270104).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-p)[ix]**2,1));c={'gain':fv.rmse(y,r007)-fv.rmse(y,p),'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist()};assert c==report['comparison' if name=='primary' else 'equal_SB_comparison'];comps[name]=c
  bins=np.empty(400,dtype=int)
  for k in range(5):bins[fid==k]=np.searchsorted(np.quantile(y[fid!=k],[.2,.8]),y[fid==k])
  tails={str(b):{'n':int((bins==b).sum()),'reference007_rmse':fv.rmse(y[bins==b],r007[bins==b]),'primary_rmse':fv.rmse(y[bins==b],d['primary'][bins==b])} for b in range(3)}
  results[split]={'scores':report['scores'],'comparisons':comps,'reload_max_error':maxerr,'train_rmse_pooled':report['train_rmse_pooled'],'foldtrain_quantile_diagnostics':tails,'prediction_sha256':report['predictions_sha256']}
 sel=json.loads((e.OUT/'selection.json').read_text());c=results['original']['comparisons']['primary'];assert sel['advance']==bool(c['gain']>c['bootstrap_sd'] and c['positive_folds']>=3)
 if not sel['advance']:assert set(results)=={'original'}
 confirmed=len(results)==3 and all(r['comparisons']['primary']['gain']>0 and r['comparisons']['primary']['positive_folds']>=3 for r in results.values());recipe=None
 if confirmed:
  mix_eligible=all(r['comparisons']['equal_SB']['gain']>0 and r['comparisons']['equal_SB']['positive_folds']>=3 for r in results.values());mix_beats=all(r['scores']['equal_SB']['rmse']<r['scores']['primary']['rmse'] for r in results.values());recipe='equal_SB' if mix_eligible and mix_beats else 'primary'
 e.check();result={'passed':True,'results':results,'models_verified':count,'full100epoch_replays':replays,'confirmed_for_final_fit':confirmed,'selected_recipe':recipe,'protected_files_verified':len(plan['protected']),'source_sha256':fv.sha(Path(__file__)),'seconds':time.perf_counter()-t,'limitations':fv.CAVEAT,'scope':'Development400 only; no holdout labels/images or test inspected; no new installs/cost/upload.'};name='dinob-mlp-063-validation.json' if not sel['advance'] or len(results)==3 else 'dinob-mlp-063-screen-validation.json';fv.write(_repo_path(ROOT, 'reports', name),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
