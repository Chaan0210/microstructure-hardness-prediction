"""Separate-process frozen spatial feature and fold-model validation."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(ROOT))
import numpy as np,joblib,torch
import dino_spatial_039 as ds
from dino_regression_037 import FrozenRegressor
import frozen_validation as fv
root=_repo_path(ROOT, 'artifacts/dino-spatial-039');out=_repo_path(root, 'regression');plan=json.loads((_repo_path(out, 'plan.json')).read_text());selection=json.loads((_repo_path(out, 'selection.json')).read_text());extract=json.loads((_repo_path(root, 'extraction.json')).read_text())
for n,h in {**plan['source_sha256'],**plan['regression_source_sha256'],**plan['feature_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
with np.load(_repo_path(ROOT, 'artifacts/dino-frozen-037/development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();panels={'control252':z['X'].astype(np.float64)}
with np.load(_repo_path(root, 'development_features.npz'),allow_pickle=False) as z:assert ids==z['ids'].tolist();panels.update({a:z[a].astype(np.float64) for a in ds.ARMS})
with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert ids==z['ids'].tolist();y=z['y'].copy();groups=z['groups'].copy()
assert len(set(ids))==len(ids)==400 and extract['weights_unchanged']
for a,X in panels.items():assert len(X)==400 and np.isfinite(X).all()
for i,h in extract['image_sha256'].items():assert i in ids and fv.sha(_repo_path(ROOT, 'data/train', f'{i}.png'))==h
# Reproduce first2rows independently onCPU;no labels select images.
torch.set_num_threads(2);model=ds.base.model();errors={}
for arm in ds.ARMS:
 a=np.stack([ds.image_features(model,i,arm,'cpu') for i in ids[:2]]);np.testing.assert_allclose(a,panels[arm][:2],rtol=2e-3,atol=2e-3);errors[arm]=float(abs(a-panels[arm][:2]).max())
del model
splits=['original']+(['20261031','20261101'] if selection['selected'] else []);results={};count=0;reloadmax=0.;squared_ref=[];squared_new=[]
for split in splits:
 d=_repo_path(out, split);r=json.loads((d/'report.json').read_text())
 with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
 assert [a['ID'] for a in rows]==ids;np.testing.assert_array_equal(y,[float(a['hardness']) for a in rows]);p={n:np.array([float(a[n]) for a in rows]) for n in r['scores']};fid=np.array([int(a['fold']) for a in rows])
 for n,a in p.items():assert np.isfinite(a).all() and abs(fv.rmse(y,a)-r['scores'][n]['rmse'])<1e-10
 arms=list(r['folds'][0]['models'])
 for a in arms:np.testing.assert_allclose(p[a+'_blend'],.75*p['reference003']+.25*p[a],atol=1e-12,rtol=0)
 for k,f in enumerate(r['folds']):
  ti=np.array([ids.index(i) for i in f['train_ids']]);vi=np.array([ids.index(i) for i in f['validation_ids']]);assert len(ti)==320 and len(vi)==80 and not set(ti)&set(vi) and not set(groups[ti])&set(groups[vi]);assert (fid[vi]==k).all()
  for it,iv in f['inner_splits']:assert not set(it)&set(iv) and set(it)|set(iv)==set(ti) and not set(groups[it])&set(groups[iv])
  for arm,rec in f['models'].items():
   path=d/f'fold-{k}'/f'{arm}.joblib';assert fv.sha(path)==rec['sha256'];obj=joblib.load(_repo_io(path));m=obj['model'];assert obj['train_ids']==f['train_ids'] and obj['validation_ids']==f['validation_ids'];X=panels[arm];np.testing.assert_allclose(m.scaler.mean_,X[ti].mean(0),rtol=0,atol=1e-10);assert m.scaler.n_samples_seen_==320 and abs(m.ymean-y[ti].mean())<1e-10 and abs(m.ystd-y[ti].std())<1e-10
   chosen=min(rec['inner_scores'],key=lambda a:a['rmse']);assert m.alpha==chosen['lambda']*X.shape[1] and rec['chosen_lambda']==chosen['lambda'];pred=m.predict(X[vi]);np.testing.assert_allclose(pred,p[arm][vi],atol=1e-10,rtol=0);reloadmax=max(reloadmax,float(abs(pred-p[arm][vi]).max()));count+=1
 s=selection['selected'];diagnostic={}
 if s:
  new=p[s+'_blend'];ref=p['reference003'];squared_ref.append((y-ref)**2);squared_new.append((y-new)**2)
  q=np.quantile(y,[.2,.8]);masks={'low20pct':y<=q[0],'middle60pct':(y>q[0])&(y<q[1]),'high20pct':y>=q[1]}
  diagnostic={'baseline_residual_correlation':float(np.corrcoef(y-ref,y-p[s])[0,1]),'by_target_range':{n:{'rows':int(mask.sum()),'baseline_rmse':fv.rmse(y[mask],ref[mask]),'new_rmse':fv.rmse(y[mask],new[mask])} for n,mask in masks.items()}}
 results[split]={'scores':r['scores'],'seconds':r['seconds'],'chosen_lambdas':{a:[f['models'][a]['chosen_lambda'] for f in r['folds']] for a in arms},'diagnostic':diagnostic,'oof_sha256':fv.sha(d/'development_oof.csv')}
scores=results['original']['scores'];eligible=[a for a in ds.ARMS if scores[a+'_blend']['comparison']['gain003']>0 and scores[a+'_blend']['comparison']['gain_matched252']>0 and sum(g>0 for g in scores[a+'_blend']['comparison']['fold_gains003'])>=3];chosen=min(eligible,key=lambda a:scores[a+'_blend']['rmse']) if eligible else None;assert chosen==selection['selected'] and eligible==selection['eligible']
cluster={}
if chosen:
 ref=np.mean(squared_ref,axis=0);new=np.mean(squared_new,axis=0);ix=np.random.default_rng(20261124).integers(0,400,(4000,400));delta=np.sqrt(ref[ix].mean(1))-np.sqrt(new[ix].mean(1));cluster={'descriptive_pooled_rmse_gain':float(np.sqrt(ref.mean())-np.sqrt(new.mean())),'conditional_id_cluster_ci95':np.percentile(delta,[2.5,97.5]).tolist(),'limits':'ResampleIDs keeping all3splitpredictions together,not1200independentobservations. Still omits model-fitting and repeated-selection uncertainty.'}
summary={'validation':{'unique_development_rows':400,'feature_cpu_reload_max_abs_errors':errors,'regression_models_reloaded':count,'max_prediction_reload_error':reloadmax,'ids_finite_metrics_fixedblend_hashes_checked':True,'fold_only_scaler_target_and_inner_split_checks':True,'fixed_selection_gate_reproduced':True,'best003_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'))},'results':results,'pooled_diagnostic':cluster,'selected':chosen,'extraction_seconds':extract['seconds'],'regression_seconds':sum(a['seconds'] for a in results.values()),'limitations':['Development400andpartitionsalreadyinformedhistoricalchoices; not independentunseenvalidation.','Spatialcropmean/std discardspositionorder andoverlapoverweightscenter; measureslocalcontext/heterogeneity, not spatialarrangement.','Featurecountscalingdoesnot equalizecorrelation/effectivecomplexity.','Whole504interpolation creates no independentpixelinformation; changes modeltokenization/relativegrain scale.','No data beyonddevelopment400,backboneadaptation,newdependencies,costorupload.']}
path=_repo_path(ROOT, 'reports/dino-039-validation.json')
if path.exists():assert json.loads(path.read_text())==summary
else:path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
