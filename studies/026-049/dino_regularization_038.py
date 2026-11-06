"""Single bounded boundary diagnostic after037 chose alpha1000 in15/15folds."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from dino_regression_037 import FrozenRegressor
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/dino-regularization-038')
ALPHAS=(1000.,3000.,10000.)
OUT.mkdir(exist_ok=False)
plan={'reason':'037 inner selection picked original upper boundaryalpha1000in15/15outerfolds; bounded higher regularization diagnostic','alphas':ALPHAS,'kind':'all1152DINOfeaturesStandardScalerRidge','inner':'GroupKFold3seed20261120+outerfold, same037partitions','blend':'0.75samefold003+0.25DINO,fixed','reference':'Reuse037fresh003OOFalreadyfitwithinexactsamefold; no model refit needed for unchanged recipe. Check037artifact hashes and exactIDs.','stop':'One predeclared expansion only; no additional grids/weights/features in038','limits':fv.CAVEAT+' Additional post037 diagnostic selection adds optimism.','source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ('dino_regularization_038.py','dino_regression_037.py')},'protected':{n:fv.sha(_repo_path(ROOT, n)) for n in ('artifacts/submission-candidate-003.csv','artifacts/final-model-017/report.json')}};fv.write(_repo_path(OUT, 'plan.json'),plan)
with np.load(_repo_path(ROOT, 'artifacts/dino-frozen-037/development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();X=z['X'].astype(np.float64)
with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert ids==z['ids'].tolist();y=z['y'].copy();groups=z['groups'].copy()
checked=json.loads((_repo_path(ROOT, 'reports/dino-037-validation.json')).read_text());allresults={};start=time.perf_counter()
for split in ('original','20261031','20261101'):
 d=_repo_path(OUT, split);d.mkdir();prior=_repo_path(ROOT, 'artifacts/dino-frozen-037/regression', split, 'development_oof.csv');assert fv.sha(prior)==checked['results'][split]['oof_sha256']
 with prior.open() as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);fid=np.array([int(r['fold']) for r in rows]);ref=np.array([float(r['fresh003']) for r in rows]);old=np.array([float(r['all_ridge_blend']) for r in rows]);pred=np.full(400,np.nan);logs=[]
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261120+k).split(X[tr],y[tr],groups[tr]));scores=[]
  for alpha in ALPHAS:
   p=np.full(len(tr),np.nan)
   for it,iv in inner:
    assert not set(groups[tr[it]])&set(groups[tr[iv]]);m=FrozenRegressor('all_ridge',alpha).fit(X[tr[it]],y[tr[it]]);p[iv]=m.predict(X[tr[iv]])
   scores.append({'alpha':alpha,'rmse':fv.rmse(y[tr],p)})
  alpha=min(scores,key=lambda r:r['rmse'])['alpha'];m=FrozenRegressor('all_ridge',alpha).fit(X[tr],y[tr]);pred[va]=m.predict(X[va]);path=d/f'fold-{k}.joblib';joblib.dump({'model':m,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(path));np.testing.assert_allclose(joblib.load(_repo_io(path))['model'].predict(X[va]),pred[va],rtol=0,atol=1e-10);np.testing.assert_allclose(m.scaler.mean_,X[tr].mean(0),rtol=0,atol=1e-10);logs.append({'fold':k,'inner_scores':scores,'alpha':alpha,'train_rmse':fv.rmse(y[tr],m.predict(X[tr])),'validation_rmse':fv.rmse(y[va],pred[va]),'sha256':fv.sha(path),'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]})
 blend=.75*ref+.25*pred;ix=np.random.default_rng(20261122).integers(0,400,(2000,400));delta=np.sqrt(np.mean((y-ref)[ix]**2,axis=1))-np.sqrt(np.mean((y-blend)[ix]**2,axis=1));report={'reference003':fv.rmse(y,ref),'old037_blend':fv.rmse(y,old),'dino_standalone':fv.rmse(y,pred),'blend':fv.rmse(y,blend),'gain':fv.rmse(y,ref)-fv.rmse(y,blend),'fold_gains':[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],blend[fid==k]) for k in range(5)],'conditional_ci95':np.percentile(delta,[2.5,97.5]).tolist(),'folds':logs}
 with (d/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold','reference003','old037_blend','dino','blend']);w.writerows(zip(ids,y,fid,ref,old,pred,blend))
 fv.write(d/'report.json',report);allresults[split]=report;print(json.dumps({'split':split,**{n:v for n,v in report.items() if n!='folds'},'alphas':[f['alpha'] for f in logs]}),flush=True)
for n,h in {**plan['source_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
fv.write(_repo_path(OUT, 'report.json'),{'results':allresults,'seconds':time.perf_counter()-start,'plan':plan})
