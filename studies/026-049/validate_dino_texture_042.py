"""Independent reload validation of completed development-only study042."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np,torch,joblib
import dino_texture_042 as dt
import frozen_validation as fv
from dino_regression_037 import FrozenRegressor
base=_repo_path(ROOT, 'artifacts/dino-texture-042');out=_repo_path(base, 'regression')
plan=json.loads((_repo_path(out, 'plan.json')).read_text());ex=json.loads((_repo_path(base, 'extraction.json')).read_text())
for n,h in {**plan['protected'],**plan['source_sha256'],**plan['regression_source_sha256']}.items():assert fv.sha(_repo_path(ROOT, n))==h
assert fv.sha(_repo_path(base, 'development_features.npz'))==plan['feature_sha256']==ex['feature_sha256']
with np.load(_repo_path(base, 'development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();X=z['X'].astype(np.float64)
with np.load(_repo_path(ROOT, 'artifacts/dino-spatial-039/development_features.npz'),allow_pickle=False) as z:assert ids==z['ids'].tolist();G=z['full504'].astype(np.float64)
with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert ids==z['ids'].tolist();y=z['y'].copy();groups=z['groups'].copy()
assert len(set(ids))==400 and X.shape==(400,1920) and np.isfinite(X).all() and ex['weights_unchanged'];np.testing.assert_array_equal(X[:,:1152],G)
for i,h in ex['image_sha256'].items():assert i in ids and fv.sha(_repo_path(ROOT, 'data/train', f'{i}.png'))==h
assert len(ex['image_sha256'])==400
# Two preselected IDs; CPU forward only, no target use or fit.
torch.set_num_threads(2);model=dt.ds.base.model()
a=np.concatenate([dt.features(model,torch.from_numpy(dt.ds.inputs(i,'full504'))).numpy() for i in ids[:2]])
np.testing.assert_allclose(a,X[:2],atol=2e-3,rtol=2e-3);featureerror=float(abs(a-X[:2]).max());del model
panels={'control005':G,'texture':X};count=0;maxerror=0.;results={}
verified039=json.loads((_repo_path(ROOT, 'reports/dino-039-validation.json')).read_text())
for split in ('original','20261031','20261101'):
 d=_repo_path(out, split);r=json.loads((d/'report.json').read_text());src=_repo_path(ROOT, 'artifacts/dino-spatial-039/regression', split, 'development_oof.csv')
 assert fv.sha(src)==r['reference_oof_sha256']==verified039['results'][split]['oof_sha256']
 with src.open() as f:old=list(csv.DictReader(f))
 with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids==[r['ID'] for r in old];np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows])
 fid=np.array([int(r['fold']) for r in rows]);p={n:np.array([float(r[n]) for r in rows]) for n in r['scores']}
 np.testing.assert_array_equal(fid,[int(r['fold']) for r in old]);np.testing.assert_array_equal(p['reference003'],[float(r['reference003']) for r in old]);np.testing.assert_allclose(p['control005_blend'],[float(r['full504_blend']) for r in old],atol=1e-10,rtol=0)
 for n,v in p.items():
  assert np.isfinite(v).all() and abs(fv.rmse(y,v)-r['scores'][n]['rmse'])<1e-10
  np.testing.assert_allclose([fv.rmse(y[fid==k],v[fid==k]) for k in range(5)],r['scores'][n]['fold_rmse'],atol=1e-10,rtol=0)
 for arm in panels:np.testing.assert_allclose(p[arm+'_blend'],.75*p['reference003']+.25*p[arm],atol=1e-12,rtol=0)
 for k,f in enumerate(r['folds']):
  ti=np.array([ids.index(i) for i in f['train_ids']]);vi=np.array([ids.index(i) for i in f['validation_ids']]);assert len(ti)==320 and len(vi)==80 and not set(ti)&set(vi) and set(ti)|set(vi)==set(range(400)) and not set(groups[ti])&set(groups[vi]);assert (fid[vi]==k).all()
  innerheld=[]
  for it,iv in f['inner_splits']:
   assert not set(it)&set(iv) and set(it)|set(iv)==set(ti) and not set(groups[it])&set(groups[iv]);innerheld.extend(iv)
  assert sorted(innerheld)==sorted(ti)
  for arm,rec in f['models'].items():
   path=d/f'fold-{k}'/f'{arm}.joblib';assert fv.sha(path)==rec['sha256'];obj=joblib.load(_repo_io(path));m=obj['model'];A=panels[arm]
   assert obj['train_ids']==f['train_ids'] and obj['validation_ids']==f['validation_ids'];np.testing.assert_allclose(m.scaler.mean_,A[ti].mean(0),rtol=0,atol=1e-10)
   assert m.scaler.n_samples_seen_==320 and abs(m.ymean-y[ti].mean())<1e-10 and abs(m.ystd-y[ti].std())<1e-10
   chosen=min(rec['inner_scores'],key=lambda v:v['rmse']);assert m.alpha==chosen['lambda']*A.shape[1] and rec['chosen_lambda']==chosen['lambda']
   pred=m.predict(A[vi]);np.testing.assert_allclose(pred,p[arm][vi],atol=1e-10,rtol=0);maxerror=max(maxerror,float(abs(pred-p[arm][vi]).max()));count+=1
   assert abs(fv.rmse(y[ti],m.predict(A[ti]))-rec['train_rmse'])<1e-10
 gain=fv.rmse(y,p['control005_blend'])-fv.rmse(y,p['texture_blend']);assert abs(gain-r['comparison']['gain_vs005'])<1e-10
 results[split]={'scores':{n:v['rmse'] for n,v in r['scores'].items()},'comparison':r['comparison'],'chosen_lambdas':{n:[f['models'][n]['chosen_lambda'] for f in r['folds']] for n in panels},'mean_train_rmse':{n:float(np.mean([f['models'][n]['train_rmse'] for f in r['folds']])) for n in panels},'texture_control_residual_correlation':float(np.corrcoef(y-p['texture'],y-p['control005'])[0,1]),'texture_003_residual_correlation':float(np.corrcoef(y-p['texture'],y-p['reference003'])[0,1]),'oof_sha256':fv.sha(d/'development_oof.csv')}
assert count==30
for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
besthash=fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-005.csv'));assert besthash=='473ef271d35b2c18185690471e2ffbada07a01cf31d07be4c2d489bd76e67ff4'
errata={'smoke_translation_invariance':'Tests additive token-value offset z+7, not spatial translation invariance.','plan_alignment_magnitude':'Averaging horizontal and vertical semivariograms discards their difference; alignment magnitude is not generally preserved. Interpret as axis-averaged spatial roughness/correlation scale.','artifact_policy':'Original hashed source and plans retained unchanged; wording corrected here. Numerical computations unchanged.'}
summary={'validation':{'development_rows':400,'shape':[400,1920],'global504_columns_exactly_equal':True,'cpu_recompute_max_feature_error':featureerror,'regression_models_reloaded':count,'max_reload_prediction_error':maxerror,'metrics_folds_groups_inner_splits_scaler_target_normalization_checked':True,'protected_artifacts_unchanged':True,'best005_sha256':besthash},'results':results,'errata':errata,'decision':'No consistent improvement over005; preserve005; no newcandidate or submission.','limitations':['All partitions reuse historically inspected development400; repeated model selection can be optimistic.','Conditional bootstrap intervals omit training and historical selection uncertainty; all include zero.','No holdout/test access, backbone adaptation, package install, external data, cost or upload in this study.','Negative result applies to this single fixed spatial pooling panel, not all DINO adaptation methods.'],'extraction_seconds':ex['seconds'],'regression_seconds':sum(json.loads((_repo_path(out, s, 'report.json')).read_text())['seconds'] for s in results)}
p=_repo_path(ROOT, 'reports/dino-042-validation.json')
if p.exists():assert json.loads(p.read_text())==summary
else:p.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
