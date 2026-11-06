"""Matched feature-count-normalized Ridge comparison for frozen spatial panels."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
from dino_regression_037 import FrozenRegressor
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/dino-spatial-039/regression')
LAMBDAS=(.1,1.,10.)
ARMS=('control252','full504','native196x4')
def prepare():
 OUT.mkdir(exist_ok=False);plan=json.loads((_repo_path(OUT.parent, 'plan.json')).read_text());plan['regression_source_sha256']={n:fv.sha(_repo_path(ROOT, n)) for n in ('dino_spatial_cv_039.py','dino_regression_037.py')};plan['feature_sha256']={n:fv.sha(_repo_path(ROOT, n)) for n in ('artifacts/dino-frozen-037/development_features.npz','artifacts/dino-spatial-039/development_features.npz')};plan['baseline_reference']='Use037validated003models/OOF trainedwithinexactmatchedfolds; no oldOOF everusedasregressortraininginputs or targets.';fv.write(_repo_path(OUT, 'plan.json'),plan)
def run(split):
 plan=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in {**plan['regression_source_sha256'],**plan['feature_sha256']}.items():assert fv.sha(_repo_path(ROOT, n))==h
 arms=ARMS if split=='original' else ('control252',json.loads((_repo_path(OUT, 'selection.json')).read_text())['selected']);assert all(a in ARMS for a in arms)
 with np.load(_repo_path(ROOT, 'artifacts/dino-frozen-037/development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();panels={'control252':z['X'].astype(np.float64)}
 with np.load(_repo_path(OUT.parent, 'development_features.npz'),allow_pickle=False) as z:assert ids==z['ids'].tolist();panels.update({a:z[a].astype(np.float64) for a in ARMS[1:]})
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert ids==z['ids'].tolist();y=z['y'].copy();groups=z['groups'].copy()
 source=_repo_path(ROOT, 'artifacts/dino-frozen-037/regression', split, 'development_oof.csv');validation=json.loads((_repo_path(ROOT, 'reports/dino-037-validation.json')).read_text());assert fv.sha(source)==validation['results'][split]['oof_sha256']
 with source.open() as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);fid=np.array([int(r['fold']) for r in rows]);ref=np.array([float(r['fresh003']) for r in rows]);dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);pred={a:np.full(400,np.nan) for a in arms};logs=[];start=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261120+k).split(panels['control252'][tr],y[tr],groups[tr]));fd=dest/f'fold-{k}';fd.mkdir();log={'fold':k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'inner_splits':[[tr[a].tolist(),tr[b].tolist()] for a,b in inner],'models':{}}
  for arm in arms:
   X=panels[arm];scores=[]
   for lam in LAMBDAS:
    p=np.full(len(tr),np.nan)
    for it,iv in inner:
     assert not set(groups[tr[it]])&set(groups[tr[iv]]);m=FrozenRegressor('all_ridge',lam*X.shape[1]).fit(X[tr[it]],y[tr[it]]);p[iv]=m.predict(X[tr[iv]])
    scores.append({'lambda':lam,'alpha':lam*X.shape[1],'rmse':fv.rmse(y[tr],p)})
   chosen=min(scores,key=lambda a:a['rmse']);m=FrozenRegressor('all_ridge',chosen['alpha']).fit(X[tr],y[tr]);pred[arm][va]=m.predict(X[va]);path=fd/f'{arm}.joblib';joblib.dump({'model':m,'train_ids':log['train_ids'],'validation_ids':log['validation_ids']},_repo_io(path));np.testing.assert_allclose(joblib.load(_repo_io(path))['model'].predict(X[va]),pred[arm][va],atol=1e-10,rtol=0);log['models'][arm]={'inner_scores':scores,'chosen_lambda':chosen['lambda'],'train_rmse':fv.rmse(y[tr],m.predict(X[tr])),'validation_rmse':fv.rmse(y[va],pred[arm][va]),'sha256':fv.sha(path)}
  logs.append(log);fv.write(dest/f'fold-{k}-report.json',log);print(json.dumps({'split':split,'fold':k,'rmse':{a:v['validation_rmse'] for a,v in log['models'].items()}}),flush=True)
 pred.update({a+'_blend':.75*ref+.25*p for a,p in list(pred.items())});pred['reference003']=ref;ix=np.random.default_rng(20261123).integers(0,400,(2000,400));scores={}
 for n,p in pred.items():scores[n]={'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]}
 for arm in arms:
  p=pred[arm+'_blend'];gain=np.sqrt(np.mean((y-ref)[ix]**2,axis=1))-np.sqrt(np.mean((y-p)[ix]**2,axis=1));scores[arm+'_blend']['comparison']={'gain003':fv.rmse(y,ref)-fv.rmse(y,p),'gain_matched252':fv.rmse(y,pred['control252_blend'])-fv.rmse(y,p),'fold_gains003':[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],p[fid==k]) for k in range(5)],'conditional_ci95':np.percentile(gain,[2.5,97.5]).tolist()}
 with (dest/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold',*pred]);w.writerows([i,y[j],fid[j],*[a[j] for a in pred.values()]] for j,i in enumerate(ids))
 report={'scores':scores,'folds':logs,'seconds':time.perf_counter()-start,'reference_oof_sha256':fv.sha(source),'limits':fv.CAVEAT};fv.write(dest/'report.json',report)
 if split=='original':
  eligible=[a for a in ARMS[1:] if scores[a+'_blend']['comparison']['gain003']>0 and scores[a+'_blend']['comparison']['gain_matched252']>0 and sum(g>0 for g in scores[a+'_blend']['comparison']['fold_gains003'])>=3];selected=min(eligible,key=lambda a:scores[a+'_blend']['rmse']) if eligible else None;fv.write(_repo_path(OUT, 'selection.json'),{'selected':selected,'eligible':eligible,'rule':plan['screen']})
 for n,h in {**plan['protected'],**plan['regression_source_sha256']}.items():assert fv.sha(_repo_path(ROOT, n))==h
 print(json.dumps({'split':split,'scores':scores,'seconds':report['seconds']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','original','20261031','20261101'));a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
