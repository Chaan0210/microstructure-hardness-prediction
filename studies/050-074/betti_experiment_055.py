"""Preregistered signed-distance Betti132 vs zero-level counts12 diagnostic."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,joblib
from PIL import Image
from sklearn.model_selection import GroupKFold
from dino_regression_037 import FrozenRegressor
import frozen_validation as fv
import betti_features_055 as f
import test_betti_055 as tests
OUT=_repo_path(ROOT, 'artifacts/betti-055');ALPHAS=(12.,120.,1200.)
def ids_only():return sorted(json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text())['development_folds'])
def image(ident):
 assert ident in ids_only()
 with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:assert im.size==(256,256) and im.mode=='L';return np.asarray(im,dtype=np.uint8).copy()
def prepare():
 OUT.mkdir(exist_ok=False)
 old=json.loads((_repo_path(ROOT, 'artifacts/lineal-054/plan.json')).read_text());protected=old['protected']
 p={'hypothesis':'Image-local signed-distance Betti curves may encode neck closure and hole/component survival beyond raw counts or granulometry. Hardness improvement unproven.','references':['https://advanced.onlinelibrary.wiley.com/doi/10.1002/adts.201900227','https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019WR025171'],'reference_access':'Wang2020 abstract and Moon2019 abstract/methods read via public publisher pages. Own 2D discrete Betti curves, NOT persistent homology, persistence diagrams/images, or paper performance reproduction. No external training data/code.','definition':'Raw grayscale image quantiles20/50/80; <= retains ties. SEDT=EDT(mask)-EDT(complement), nearest opposite pixel inside image. Allone +inf/allzero -inf explicitly. Excursion SEDT>=d at -16,-8,-4,-2,-1,0,1,2,4,8,16 pixels. Finite rectangle/no wrap; topological outside is background. Foreground4/background8 or foreground8/background4, beta0=components and beta1=beta0-Euler. Edge-connected background never a hole. Each count divided by original image area65536. 3*11*2*2=132.','boundary_limit':'Observed-image topology; structures truncated by edges remain censored. Distance does not invent opposite phase outside. d0 and d1 may duplicate because signed pixel distances have magnitude>=1.','preprocessing':'Raw images no denoise; all X scaler and targetmean/std fit inside each innertrain/outertrain. No residual correction.','regression':'Ridge alpha12/120/1200, innerGroupKFold3shuffle seed20261223+fold; inner pooled standalone RMSE selects alpha. Primary=.9*samefold007+.1*Betti132.','diagnostic':'Zero-distance12 counts use same innerRidge grid; standalone and fixed.1 blend only diagnostic, not eligible alternatewinner. No feature/weight tuning.','screen':'Original gain>conditionalpairedbootstrapSD AND >=3/5 positive folds. Compute allocation only, not significance. Only then both existing repartitions, unchanged design.','diagnostic_plan':'Feature constants/duplicates; betti curves vs zero counts using foldtrain Ridge reconstruction (alpha120 fixed); image highfrequencyRMS association descriptive only (contains edges,not noise estimate); outertrain feature range and prediction range extrapolation diagnostics. No new features chosen. Granulometry differs: retained intensity/area vs topology; fixture checks equal raw counts but differing erosion connectivity.','scope':'development400 only; no holdout/test data, candidateCSV/upload, installs/cost. Best007 protected.','bias':fv.CAVEAT,'resource':'Fixtures then10images; projected400<300s andRSS<2GB before full extraction','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['betti_features_055.py','test_betti_055.py','betti_experiment_055.py','dino_regression_037.py']},'protected':protected}
 fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'features':132}))
def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,d in {**p['sources'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==d,n
 return p
def benchmark():
 check();fixture=tests.run();fv.write(_repo_path(OUT, 'fixtures.json'),fixture);t=time.perf_counter();ids=ids_only()[:10];X=np.array([f.extract(image(i))[0] for i in ids]);seconds=time.perf_counter()-t
 r={'n':10,'seconds':seconds,'projected400_seconds':seconds*40,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};r['resource_pass']=bool(r['projected400_seconds']<300 and r['peak_rss_bytes']<2e9)
 with (_repo_path(OUT, 'benchmark_features.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),X=X)
 fv.write(_repo_path(OUT, 'benchmark.json'),r);print(json.dumps({'fixtures':fixture,**r}),flush=True)
def extract():
 check();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass'];t=time.perf_counter();ids=ids_only();X=[];metadata=[];hashes={}
 for j,ident in enumerate(ids):
  a,m=f.extract(image(ident));X.append(a);metadata.append(m);hashes[ident]=fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))
  if (j+1)%80==0:print(json.dumps({'features_done':j+1,'seconds':time.perf_counter()-t}),flush=True)
 X=np.array(X);assert X.shape==(400,132) and np.isfinite(X).all()
 with np.load(_repo_path(OUT, 'benchmark_features.npz')) as z:np.testing.assert_array_equal(z['X'],X[:10])
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),X=X,names=np.array(f.NAMES))
 r={'seconds':time.perf_counter()-t,'feature_sha256':fv.sha(_repo_path(OUT, 'development_features.npz')),'image_sha256':hashes,'metadata':metadata,'shape':list(X.shape)};fv.write(_repo_path(OUT, 'extraction.json'),r);print(json.dumps({'extraction_seconds':r['seconds']}),flush=True)
def run(split):
 check()
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 ex=json.loads((_repo_path(OUT, 'extraction.json')).read_text());assert fv.sha(_repo_path(OUT, 'development_features.npz'))==ex['feature_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz')) as z:ids=z['ids'].tolist();X=z['X'].copy()
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz')) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 src=_repo_path(ROOT, 'artifacts/seed-stability-052', split, 'development_oof.csv');verified=json.loads((_repo_path(ROOT, 'reports/seed-052-validation.json')).read_text());assert fv.sha(src)==verified['results'][split]['oof_sha256']
 with src.open() as file:rows=list(csv.DictReader(file))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);fid=np.array([int(r['fold']) for r in rows]);ref=np.array([float(r['candidate007_design']) for r in rows]);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);t=time.perf_counter();pred={'reference007':ref};logs=[]
 for arm,cols in [('betti132',np.arange(132)),('counts12',f.ZERO_COLUMNS)]:
  A=X[:,cols];p=np.full(400,np.nan)
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261223+k).split(A[tr],y[tr],groups[tr]));scores=[]
   for alpha in ALPHAS:
    ip=np.full(len(tr),np.nan)
    for it,iv in inner:
     assert not set(groups[tr[it]])&set(groups[tr[iv]]);m=FrozenRegressor('all_ridge',alpha).fit(A[tr[it]],y[tr[it]]);ip[iv]=m.predict(A[tr[iv]])
    scores.append({'alpha':alpha,'rmse':fv.rmse(y[tr],ip)})
   chosen=min(scores,key=lambda a:a['rmse']);m=FrozenRegressor('all_ridge',chosen['alpha']).fit(A[tr],y[tr]);p[va]=m.predict(A[va]);path=out/f'{arm}-fold-{k}.joblib';joblib.dump({'model':m,'columns':cols,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(path))
   logs.append({'arm':arm,'fold':k,'chosen':chosen,'inner_scores':scores,'inner_splits':[[tr[a].tolist(),tr[b].tolist()] for a,b in inner],'train_rmse':fv.rmse(y[tr],m.predict(A[tr])),'validation_rmse':fv.rmse(y[va],p[va]),'model_sha256':fv.sha(path),'validation_outside_train_label_range':int(((p[va]<y[tr].min())|(p[va]>y[tr].max())).sum()),'validation_feature_cells_outside_train_range_fraction':float(((A[va]<A[tr].min(0))|(A[va]>A[tr].max(0))).mean()),'prediction_range':[float(p[va].min()),float(p[va].max())]})
  pred[arm]=p;pred[arm+'_blend']=.9*ref+.1*p
 scores={n:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for n,v in pred.items()};new=pred['betti132_blend'];gain=fv.rmse(y,ref)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20261224).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-ref)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));comp={'gain':gain,'fold_gains':fg,'positive_folds':sum(v>0 for v in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'residual_correlation':float(np.corrcoef(y-ref,y-pred['betti132'])[0,1])}
 with (out/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),y=y,fold=fid,**pred)
 report={'split':split,'scores':scores,'comparison':comp,'folds':logs,'seconds':time.perf_counter()-t,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'reference_oof_sha256':fv.sha(src),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>boot.std() and sum(v>0 for v in fg)>=3),'primary':'betti132_blend','comparison':comp,'diagnostic_ineligible':True})
 check();print(json.dumps({k:v for k,v in report.items() if k!='folds'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','extract','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark,'extract':extract}.get(a.mode,lambda:run(a.mode))()
