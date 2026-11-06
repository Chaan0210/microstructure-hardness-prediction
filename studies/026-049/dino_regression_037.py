"""Frozen DINOv2 nested regression, predetermined small originalCV screen."""
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
import numpy as np
import torch,joblib
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupKFold
import frozen_validation as fv
import cnn_structure_experiment as core
OUT=_repo_path(ROOT, 'artifacts/dino-frozen-037/regression')
ARMS=('cls_ridge','all_ridge','all_pca64_rbf')
GRIDS={'cls_ridge':(10.,100.,1000.),'all_ridge':(10.,100.,1000.),'all_pca64_rbf':(.01,.1,1.)}
class FrozenRegressor:
 def __init__(self,kind,alpha):self.kind=kind;self.alpha=alpha
 def select(self,X):return X[:,:384] if self.kind=='cls_ridge' else X
 def fit(self,X,y):
  self.scaler=StandardScaler().fit(self.select(X));Z=self.scaler.transform(self.select(X));self.ymean=float(y.mean());self.ystd=float(y.std());assert self.ystd>0
  self.pca=None
  if self.kind=='all_pca64_rbf':
   self.pca=PCA(n_components=64,svd_solver='full').fit(Z);Z=self.pca.transform(Z);dist=pairwise_distances(Z,squared=True);med=float(np.median(dist[np.triu_indices(len(Z),1)]));assert med>0;self.gamma=1/(2*med);self.regressor=KernelRidge(alpha=self.alpha,kernel='rbf',gamma=self.gamma)
  else:self.regressor=Ridge(alpha=self.alpha)
  self.regressor.fit(Z,(y-self.ymean)/self.ystd);return self
 def predict(self,X):
  Z=self.scaler.transform(self.select(X));Z=self.pca.transform(Z) if self.pca is not None else Z
  return self.regressor.predict(Z)*self.ystd+self.ymean

# Stable import path for portable joblib artifacts, including CLI execution.
FrozenRegressor.__module__='dino_regression_037'
if __name__=='__main__':sys.modules['dino_regression_037']=sys.modules[__name__]

def prepare():
 OUT.mkdir(exist_ok=False)
 plan={'arms':list(ARMS),'grids':GRIDS,'features':'FrozenCLS384 versusCLS+patchmean+patchstd1152 from full252RGB; no backbone adaptation;noTTA','preprocessing':'StandardScaler then optionalPCA64(whitenFalse,fullSVD); all scaler/PCA/Ridge/kernel/targetmeanstd fit innertrain for selection and outertrain for finalfit','kernel_bandwidth':'gamma=1/(2*median squared distance between training PCA features), recomputed inside each fit','inner':'GroupKFold3shuffle,seed20261120+outerfold; pooled innerOOFRMSE chooses alpha','screening':'Only originalCV. Eligible if fixedblend improves003 and>=3/5folds improve; choose lowest blendRMSE; at most one arm repeated on both predefined repartitions. If noeligiblearm stop.','ensemble':'0.75*fresh003+0.25*DINO prediction; fixed weights, nopublic tuning','reference':'003 baseline+compactkernel+Tiny128CNN freshly refit everyfold; compare against saved003 to check drift','data_scope':fv.CAVEAT,'source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ('dino_regression_037.py','dino_frozen_037.py','cnn_structure_experiment.py','cnn_controls.py','frozen_validation.py','compact_regression.py')},'feature_sha256':fv.sha(_repo_path(ROOT, 'artifacts/dino-frozen-037/development_features.npz')),'protected':{n:fv.sha(_repo_path(ROOT, n)) for n in ('artifacts/submission-candidate-003.csv','artifacts/final-model-017/report.json')}}
 fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps(plan),flush=True)

def nested(kind,X,y,groups,tr,va,fold):
 split=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261120+fold).split(X[tr],y[tr],groups[tr]));scores=[]
 for alpha in GRIDS[kind]:
  p=np.full(len(tr),np.nan)
  for it,iv in split:
   assert not set(groups[tr[it]])&set(groups[tr[iv]])
   m=FrozenRegressor(kind,alpha).fit(X[tr[it]],y[tr[it]]);p[iv]=m.predict(X[tr[iv]])
  scores.append({'alpha':alpha,'inner_rmse':fv.rmse(y[tr],p)})
 chosen=min(scores,key=lambda r:r['inner_rmse'])['alpha'];m=FrozenRegressor(kind,chosen).fit(X[tr],y[tr]);return m.predict(X[va]),m,{'inner_scores':scores,'chosen_alpha':chosen,'inner_train_ids_indices':[[tr[a].tolist(),tr[b].tolist()] for a,b in split]}

def run(split):
 torch.set_num_threads(2);plan=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert torch.backends.mps.is_available()
 for n,h in plan['source_sha256'].items():assert fv.sha(_repo_path(ROOT, n))==h
 assert fv.sha(_repo_path(ROOT, 'artifacts/dino-frozen-037/development_features.npz'))==plan['feature_sha256']
 arms=ARMS if split=='original' else (json.loads((_repo_path(OUT, 'selection.json')).read_text())['selected'],);assert all(a in ARMS for a in arms)
 d=_repo_path(OUT, split);d.mkdir(exist_ok=False);ids,y,groups,Xbase,recipes,weights,panel,images,provenance=core.load_data();img=images[128];del images
 with np.load(_repo_path(ROOT, 'artifacts/dino-frozen-037/development_features.npz'),allow_pickle=False) as z:assert ids==z['ids'].tolist();X=z['X'].copy().astype(np.float64)
 sp=json.loads((_repo_path(ROOT, 'artifacts/cnn-structure-035', split, 'split_manifest.json')).read_text());fid=np.array([sp['development_folds'][i] for i in ids]);fv.write(d/'split_manifest.json',sp);fv.write(d/'input-provenance.json',provenance)
 with (_repo_path(ROOT, 'artifacts/cnn-structure-035', split, 'development_oof.csv')).open() as f:prior=list(csv.DictReader(f))
 assert [a['ID'] for a in prior]==ids;np.testing.assert_array_equal(y,[float(a['hardness']) for a in prior]);ref=np.array([float(a['recorded003']) for a in prior])
 pred={n:np.full(400,np.nan) for n in ('fresh003',*arms,*[a+'_blend' for a in arms])};logs=[];start=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va]);fd=d/f'fold-{k}';fd.mkdir();t=time.perf_counter()
  fitted={n:clone(m).fit(Xbase[tr],y[tr]) for n,m in recipes.items()};bp=sum(weights[n]*m.predict(Xbase[va]) for n,m in fitted.items());joblib.dump({'models':fitted,'weights':weights,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(fd/'baseline.joblib'));del fitted
  kp,km,kd=fv.kernel_fold(panel['X'],y,groups,panel['blocks'],tr,va,k);joblib.dump({'model':km,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(fd/'compact-kernel.joblib'));del km
  cp,cr=core.train_fold('tiny128',img,panel['X'],y,tr,va,ids,k,'mps',fd);old=.5*bp+.25*kp+.25*cp;np.testing.assert_allclose(old,ref[va],rtol=0,atol=2e-4);pred['fresh003'][va]=old
  log={'fold':k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'cnn_reference':cr,'reference003_max_abs_drift':float(abs(old-ref[va]).max()),'models':{}}
  for arm in arms:
   p,m,rec=nested(arm,X,y,groups,tr,va,k);pred[arm][va]=p;pred[arm+'_blend'][va]=.75*old+.25*p;joblib.dump({'model':m,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(fd/f'{arm}.joblib'));rec.update(train_rmse=fv.rmse(y[tr],m.predict(X[tr])),validation_rmse=fv.rmse(y[va],p));np.testing.assert_allclose(joblib.load(_repo_io(fd/f'{arm}.joblib'))['model'].predict(X[va]),p,rtol=0,atol=1e-10);log['models'][arm]=rec;print(json.dumps({'split':split,'fold':k,'arm':arm,'rmse':rec['validation_rmse'],'blend_rmse':fv.rmse(y[va],pred[arm+'_blend'][va])}),flush=True)
  log['seconds']=time.perf_counter()-t;log['sha256']={p.name:fv.sha(p) for p in fd.iterdir()};logs.append(log);fv.write(d/f'fold-{k}-report.json',log)
 scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in pred.items()};ix=np.random.default_rng(20261121).integers(0,400,(2000,400))
 for arm in arms:
  p=pred[arm+'_blend'];a=pred['fresh003'];delta=np.sqrt(np.mean((y-a)[ix]**2,axis=1))-np.sqrt(np.mean((y-p)[ix]**2,axis=1));scores[arm+'_blend']['comparison']={'gain':fv.rmse(y,a)-fv.rmse(y,p),'fold_gains':[fv.rmse(y[fid==k],a[fid==k])-fv.rmse(y[fid==k],p[fid==k]) for k in range(5)],'conditional_ci95':np.percentile(delta,[2.5,97.5]).tolist()}
 with (d/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold',*pred]);w.writerows([i,y[j],fid[j],*[a[j] for a in pred.values()]] for j,i in enumerate(ids))
 for n,h in {**plan['source_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
 report={'scores':scores,'folds':logs,'elapsed_seconds':time.perf_counter()-start,'limits':fv.CAVEAT};fv.write(d/'report.json',report)
 if split=='original':
  eligible=[a for a in arms if scores[a+'_blend']['comparison']['gain']>0 and sum(g>0 for g in scores[a+'_blend']['comparison']['fold_gains'])>=3];chosen=min(eligible,key=lambda a:scores[a+'_blend']['rmse']) if eligible else None;fv.write(_repo_path(OUT, 'selection.json'),{'selected':chosen,'eligible':eligible,'rule':plan['screening']})
 print(json.dumps({'split':split,'scores':scores,'seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','original','20261031','20261101'));a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
