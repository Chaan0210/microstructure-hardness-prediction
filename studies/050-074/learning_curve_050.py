"""Single-setting, single-order-seed learning curve. No candidate selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
from dino_regression_037 import FrozenRegressor
OUT=_repo_path(ROOT, 'artifacts/learning-curve-050')
SEED=20261127
ALPHA=3456.0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-p)**2)))

def main():
 OUT.mkdir(exist_ok=False)
 source=_repo_path(ROOT, 'artifacts/dino-multilayer-044/development_features.npz')
 labels=_repo_path(ROOT, 'artifacts/run-012/development_features.npz')
 manifest=_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')
 protected={n:sha(_repo_path(ROOT, n)) for n in ['artifacts/submission-candidate-006.csv','artifacts/dino-multilayer-044/candidate-design.json','artifacts/final-model-045/dino-regressor.joblib']}
 plan={'purpose':'Data-size/overfit/representation diagnostic, not model or score selection. No followup sweep authorized by these results.','scope':'Development400 cached frozen006 DINO3456 branch ONLY; this is not the full006 ensemble learning curve. No images, holdout/test, new features/install/cost/upload.','equivalent_previous_curve':'No equivalent nested sample-size curve found. Prior044 logs contain full320 training/validation error only; historical CNN epoch curves are different diagnostics.','outer':'Original five fixed grouped folds from run001; every validation80 held fixed at all three sample sizes.','subsample':'One seed20261127 reused to shuffle sorted320 outer-train indices separately in each fold. Prefix80 subset prefix160 subset320; group disjointness verified. No target stratification or target-aware subset selection.','fractions':[.25,.5,1.0],'train_sizes':[80,160,320],'condition':'Existing FrozenRegressor all_ridge on unchanged3456-dimensional frozen DINO cache; lambda1, alpha3456 fixed at all sizes. Middle of prior grid; no innerCV or tuning on this curve.','preprocessing':'X StandardScaler and targetmean/std fit only the current nested training subset; no PCA.','regularization_caveat':'sklearn Ridge uses summed squared loss+alpha*L2. Fixed alpha means alpha/n decreases with n. This standard fixed-hyperparameter learning curve does not isolate sample size from effective per-sample regularization.','metrics':'Pooled heldout400 RMSE, five fold validation RMSE; pooled resubstitution RMSE across5*n fit rows, optimism gap; constant subset-target-mean validation baseline. No significance or noise-floor conclusion.','selection_bias':'Previously reused development data, one subset order, overlapping fold-training sets; fold variability is descriptive, not independent replication.','baseline_comparison':'100% fixed-lambda curve differs from original006 branch inner-selected lambda on folds0/2(.1). Folds1/3/4 chosenlambda1 should reproduce saved predictions exactly. Do not compare standalonebranch RMSE to full00613.4232 as a degradation.','source_sha256':{n:sha(_repo_path(ROOT, n)) for n in ['learning_curve_050.py','dino_regression_037.py']},'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [source,labels,manifest]},'protected':protected}
 save(_repo_path(OUT, 'plan.json'),plan)
 checked=json.loads((_repo_path(ROOT, 'artifacts/dino-multilayer-044/extraction.json')).read_text());assert sha(source)==checked['feature_sha256']
 with np.load(_repo_io(source),allow_pickle=False) as z:ids=z['ids'].tolist();X=z['X'].astype(float).reshape(400,3456)
 with np.load(_repo_io(labels),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 assert len(np.unique(groups))==400
 sp=json.loads(manifest.read_text());assert set(ids)==set(sp['development_folds']);fid=np.array([sp['development_folds'][i] for i in ids]);assert set(fid)==set(range(5))
 # Load only prior development predictions/reports; no full500 estimator opened.
 prior_path=_repo_path(ROOT, 'artifacts/dino-multilayer-044/original/development_oof.csv')
 checked=json.loads((_repo_path(ROOT, 'reports/dino-044-validation.json')).read_text());assert sha(prior_path)==checked['results']['original']['oof_sha256']
 with prior_path.open() as f:prior=list(csv.DictReader(f))
 assert [r['ID'] for r in prior]==ids;assert np.array_equal(y,[float(r['hardness']) for r in prior]);assert np.array_equal(fid,[int(r['fold']) for r in prior])
 pred={n:np.full(400,np.nan) for n in (80,160,320)};constant={n:np.full(400,np.nan) for n in pred};trainres={n:[] for n in pred};records=[];t=time.perf_counter();max_reload=0.;max_prior=0.
 for fold in range(5):
  train=np.flatnonzero(fid!=fold);val=np.flatnonzero(fid==fold);assert len(train)==320 and len(val)==80
  order=np.random.default_rng(SEED).permutation(train)
  prior_sub=None
  for n in pred:
   sub=order[:n];assert not set(groups[sub])&set(groups[val])
   if prior_sub is not None:assert set(prior_sub)<=set(sub)
   prior_sub=sub.copy();model=FrozenRegressor('all_ridge',ALPHA).fit(X[sub],y[sub]);vp=model.predict(X[val]);tp=model.predict(X[sub]);pred[n][val]=vp;constant[n][val]=model.ymean;trainres[n].extend((tp-y[sub]).tolist())
   np.testing.assert_allclose(model.scaler.mean_,X[sub].mean(0),rtol=0,atol=1e-12);np.testing.assert_allclose(model.scaler.var_,X[sub].var(0),rtol=1e-12,atol=1e-12);assert int(model.scaler.n_samples_seen_)==n;assert model.ymean==float(y[sub].mean());assert model.ystd==float(y[sub].std());assert model.pca is None
   file=_repo_path(OUT, f'fold-{fold}-n{n}.joblib');joblib.dump({'model':model,'train_ids':[ids[j] for j in sub],'validation_ids':[ids[j] for j in val]},_repo_io(file));error=float(abs(joblib.load(_repo_io(file))['model'].predict(X[val])-vp).max());assert error<1e-10;max_reload=max(max_reload,error)
   if n==320 and fold in (1,3,4):
    error=float(abs(vp-np.array([float(prior[j]['multi3']) for j in val])).max());assert error<1e-9;max_prior=max(max_prior,error)
   record={'fold':fold,'n_train':n,'n_validation':len(val),'train_indices':sub.tolist(),'validation_indices':val.tolist(),'train_rmse':rmse(y[sub],tp),'validation_rmse':rmse(y[val],vp),'constant_validation_rmse':rmse(y[val],constant[n][val]),'target_mean':model.ymean,'target_std':model.ystd,'model_sha256':sha(file)};records.append(record)
 points=[]
 for n in pred:
  assert np.isfinite(pred[n]).all();r=[a for a in records if a['n_train']==n];vr=rmse(y,pred[n]);tr=float(np.sqrt(np.mean(np.array(trainres[n])**2)))
  points.append({'n_train':n,'fraction':n/320,'pooled_train_rmse':tr,'pooled_validation_rmse':vr,'generalization_gap':vr-tr,'fold_train_rmse':[a['train_rmse'] for a in r],'fold_validation_rmse':[a['validation_rmse'] for a in r],'constant_validation_rmse':rmse(y,constant[n]),'fold_validation_range':[min(a['validation_rmse'] for a in r),max(a['validation_rmse'] for a in r)]})
 differences=[]
 for low,high in [(80,160),(160,320),(80,320)]:
  a=points[[p['n_train'] for p in points].index(low)];b=points[[p['n_train'] for p in points].index(high)];g=np.array(a['fold_validation_rmse'])-b['fold_validation_rmse'];differences.append({'from_n':low,'to_n':high,'pooled_validation_gain':a['pooled_validation_rmse']-b['pooled_validation_rmse'],'fold_validation_gains':g.tolist(),'folds_improving':int((g>0).sum()),'fold_gain_mean':float(g.mean()),'fold_gain_sd':float(g.std(ddof=1))})
 with (_repo_path(OUT, 'validation_predictions.csv')).open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold','n80','n160','n320']);w.writerows([i,y[j],fid[j],*[pred[n][j] for n in pred]] for j,i in enumerate(ids))
 np.savez_compressed(_repo_path(OUT, 'training_residuals.npz'),**{f'n{n}':np.array(v) for n,v in trainres.items()})
 for path,h in {**plan['source_sha256'],**plan['input_sha256'],**protected}.items():assert sha(_repo_path(ROOT, path))==h,path
 report={'status':'completed','purpose':plan['purpose'],'scope':plan['scope'],'points':points,'differences':differences,'records':records,'seconds':time.perf_counter()-t,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'reload_max_error':max_reload,'matching_existing_full_subset_prediction_max_error':max_prior,'scaler_target_subset_only_verified':True,'nested_subsets_verified':True,'protected_unchanged':True,'new_candidate':None,'selection_performed':False,'warnings':[plan['regularization_caveat'],plan['selection_bias'],plan['baseline_comparison']]}
 save(_repo_path(OUT, 'report.json'),report);save(_repo_path(ROOT, 'reports/learning-curve-050-summary.json'),report);print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
