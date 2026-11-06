"""One spatial texture hypothesis, matched005 control in all3fixed partitions."""
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
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
from dino_regression_037 import FrozenRegressor
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/dino-texture-042/regression')
LAMBDAS=(.1,1.,10.)
def run():
 OUT.mkdir(exist_ok=False);plan=json.loads((_repo_path(OUT.parent, 'plan.json')).read_text());plan['regression_source_sha256']={n:fv.sha(_repo_path(ROOT, n)) for n in ('dino_texture_cv_042.py','dino_regression_037.py')};plan['feature_sha256']=fv.sha(_repo_path(OUT.parent, 'development_features.npz'));fv.write(_repo_path(OUT, 'plan.json'),plan)
 with np.load(_repo_path(OUT.parent, 'development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();texture=z['X'].astype(np.float64)
 with np.load(_repo_path(ROOT, 'artifacts/dino-spatial-039/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;global_=z['full504'].astype(np.float64)
 np.testing.assert_allclose(texture[:,:1152],global_,atol=2e-4,rtol=0);panels={'control005':global_,'texture':texture}
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 checked=json.loads((_repo_path(ROOT, 'reports/dino-039-validation.json')).read_text());allresults={};start=time.perf_counter()
 for split in ('original','20261031','20261101'):
  d=_repo_path(OUT, split);d.mkdir();source=_repo_path(ROOT, 'artifacts/dino-spatial-039/regression', split, 'development_oof.csv');assert fv.sha(source)==checked['results'][split]['oof_sha256']
  with source.open() as f:rows=list(csv.DictReader(f))
  assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);fid=np.array([int(r['fold']) for r in rows]);ref003=np.array([float(r['reference003']) for r in rows]);ref005=np.array([float(r['full504_blend']) for r in rows]);pred={a:np.full(400,np.nan) for a in panels};logs=[];t=time.perf_counter()
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261120+k).split(global_[tr],y[tr],groups[tr]));fd=d/f'fold-{k}';fd.mkdir();log={'fold':k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'inner_splits':[[tr[a].tolist(),tr[b].tolist()] for a,b in inner],'models':{}}
   for arm,X in panels.items():
    scores=[]
    for lam in LAMBDAS:
     p=np.full(len(tr),np.nan)
     for it,iv in inner:
      assert not set(groups[tr[it]])&set(groups[tr[iv]]);m=FrozenRegressor('all_ridge',lam*X.shape[1]).fit(X[tr[it]],y[tr[it]]);p[iv]=m.predict(X[tr[iv]])
     scores.append({'lambda':lam,'rmse':fv.rmse(y[tr],p)})
    chosen=min(scores,key=lambda a:a['rmse'])['lambda'];m=FrozenRegressor('all_ridge',chosen*X.shape[1]).fit(X[tr],y[tr]);pred[arm][va]=m.predict(X[va]);path=fd/f'{arm}.joblib';joblib.dump({'model':m,'train_ids':log['train_ids'],'validation_ids':log['validation_ids']},_repo_io(path));np.testing.assert_allclose(joblib.load(_repo_io(path))['model'].predict(X[va]),pred[arm][va],atol=1e-10,rtol=0);log['models'][arm]={'chosen_lambda':chosen,'inner_scores':scores,'train_rmse':fv.rmse(y[tr],m.predict(X[tr])),'validation_rmse':fv.rmse(y[va],pred[arm][va]),'sha256':fv.sha(path)}
   logs.append(log);fv.write(d/f'fold-{k}-report.json',log)
  pred.update({a+'_blend':.75*ref003+.25*p for a,p in list(pred.items())});pred['reference003']=ref003;np.testing.assert_allclose(pred['control005_blend'],ref005,rtol=0,atol=1e-10)
  scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in pred.items()};new=pred['texture_blend'];ref=pred['control005_blend'];ix=np.random.default_rng(20261126).integers(0,400,(2000,400));delta=np.sqrt(np.mean((y-ref)[ix]**2,axis=1))-np.sqrt(np.mean((y-new)[ix]**2,axis=1));comparison={'gain_vs005':fv.rmse(y,ref)-fv.rmse(y,new),'fold_gains':[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)],'conditional_ci95':np.percentile(delta,[2.5,97.5]).tolist()}
  with (d/'development_oof.csv').open('x',newline='') as f:
   w=csv.writer(f);w.writerow(['ID','hardness','fold',*pred]);w.writerows([i,y[j],fid[j],*[a[j] for a in pred.values()]] for j,i in enumerate(ids))
  report={'scores':scores,'comparison':comparison,'folds':logs,'seconds':time.perf_counter()-t,'reference_oof_sha256':fv.sha(source)};fv.write(d/'report.json',report);allresults[split]=report;print(json.dumps({'split':split,'scores':{n:a['rmse'] for n,a in scores.items()},'comparison':comparison,'seconds':report['seconds']}),flush=True)
 for n,h in {**plan['protected'],**plan['source_sha256'],**plan['regression_source_sha256']}.items():assert fv.sha(_repo_path(ROOT, n))==h
 fv.write(_repo_path(OUT, 'report.json'),{'results':allresults,'seconds':time.perf_counter()-start,'scope':'Development400only; allregressionpreprocessingtrainfoldonly; no test/publicscoreselection;003samefoldfixed reference reused;005controlrefitandverified'})
if __name__=='__main__':run()
