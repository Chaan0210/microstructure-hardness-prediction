"""Fixed official second-order scattering control; development400 only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import sys,json,time,hashlib,datetime,argparse,resource
from pathlib import Path
import numpy as np
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from kymatio.scattering2d.frontend.numpy_frontend import ScatteringNumPy2D as Scattering2D
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
OUT=_repo_path(ROOT, 'artifacts/wavelet-scattering-096')
SPLITS=('original','20261031','20261101');ALPHAS=(.01,.1,1.);DIMS={'order1':49,'order2':1009}
FIXED5=('TRAIN_000438','TRAIN_000331','TRAIN_000227','TRAIN_000071','TRAIN_000168')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def emit(obj):print(json.dumps(obj),flush=True)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def scope():
 m=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));ids=sorted(m['development_folds']);assert len(ids)==400 and not set(ids)&set(m['holdout']);return ids,m

def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for name,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, name))==h,name
 return p

def inputs(split):
 p=check();ids,m=scope();v=read(_repo_path(ROOT, 'reports/frozen-complement-085-validation.json'));path=_repo_path(ROOT, f'artifacts/frozen-complement-085/{split}/development_predictions.npz')
 assert sha(path)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:
  assert z['ids'].tolist()==ids;y=z['y'].copy();fid=z['fold'].copy();ref=z['primary'].copy()
 g=np.array([m['group_by_id'][i] for i in ids]);assert np.isfinite(y).all() and np.isfinite(ref).all()
 if split=='original':np.testing.assert_array_equal(fid,[m['development_folds'][i] for i in ids])
 for k in range(5):assert np.sum(fid==k)==80 and not set(g[fid==k])&set(g[fid!=k])
 assert abs(rmse(y,ref)-v['results'][split]['scores']['primary']['rmse'])<1e-10
 return ids,y,fid,ref,g

def prepare():
 smoke=read(_repo_path(OUT, 'smoke.json'));assert smoke['passed'] and smoke['resource_pass']
 env=read(_repo_path(OUT, 'environment-after.json'));assert not env['changed_or_removed']
 pub=read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'));v=read(_repo_path(ROOT, 'reports/frozen-complement-085-validation.json'));assert v['passed']
 ids,m=scope()
 sources=['wavelet_scattering_096.py','validate_wavelet_scattering_096.py','scattering_smoke_096_2d.py']
 kn=_repo_path(ROOT, '.venv/lib/python3.13/site-packages/kymatio')
 sources += [str(f.relative_to(ROOT)) for f in sorted(kn.rglob('*.py'))]
 inp=['artifacts/run-001/split_manifest.json','reports/frozen-complement-085-validation.json','reports/candidate011-public-submission.json','artifacts/wavelet-scattering-096/environment-before.json','artifacts/wavelet-scattering-096/environment-after.json','artifacts/wavelet-scattering-096/smoke.json']
 inp += [f'artifacts/frozen-complement-085/{s}/development_predictions.npz' for s in SPLITS]
 p={'study':'096 fixed second-order wavelet scattering versus first-order control','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'hypothesis':'Cascaded directional Morlet convolution and modulus add cross-scale/orientation envelope structure beyond Fourier/two-point descriptors and existing LoG tile-energy correlations. Complementarity to011 is unproven.',
 'literature':{'url':'https://onlinelibrary.wiley.com/doi/full/10.1002/pamm.202300169','scope':'256px morphology reconstruction evidence only, not hardness performance evidence.'},
 'features':'Unmodified official Kymatio0.3.0 NumPy2D CPU class; native256 grayscale float32/255, no resizing/centering/denoising/contrast normalization. J6,L8,max_order2,reflect padding. Mean of each4x4 output in float64. Keep all scales and orientations separate. No scale/rotation invariance augmentation. No feature/PCA selection. 1009=1+48+960. Control uses exact first49 coefficients (0th+1st), verified against max_order1.',
 'regression':'Each arm StandardScaler -> RBF KernelRidge gamma=1/dimension; target standardized with training-only mean/std and restored for predictions. Alpha selected only among [.01,.1,1.] using same3-fold GroupKFold(shuffle=True,seed20270320+outerfold) inside each outertrain. Each innerfit refits its scaler and targetstats. Min pooled innerOOF MSE; ties choose first. Outerfit refits chosen recipe on320. No other choices/calibration/ratios.',
 'alphas':list(ALPHAS),'gamma_rule':'1/dimension','inner_seed':20270320,'blend_weight':.1,
 'single_blend':'For each arm .9*samefold verified011 + .1*arm prediction; primary is order2 blend. BaselineOOF only enters heldout comparison and fixed combination, never regression fitting.',
 'original_gate':'Order2 blend improves011 pooledRMSE by>=.02 and>=3/5 folds; order2 standalone AND blend strictly beat pairedorder1; fixed395 excluding original093 IDs also improves011. Otherwise reject this fixed trial without resplits or first-order fallback.',
 'resplit_gate':'If original passes, run BOTH existing resplits unchanged. Each must meet the same originalgate. Allthree gates and independent validation required for eligibility. Same400 reused, not independent datasets.',
 'resource_gate':'CPU2, peakRSS<2GB; smoke projected400<1800s. Real-image extraction hard cap1800s. No other downloads or installs. No full500/test/candidate/upload in this study; report eligible result toparent if allcriteria pass.',
 'fixed5_exclusions':list(FIXED5),'development_ids':ids,'sources':{n:sha(_repo_path(ROOT, n)) for n in sources},'inputs':{n:sha(_repo_path(ROOT, n)) for n in inp},'protected011':pub['protected_sha256'],
 'bias':'One prespecified candidate on repeatedly reused400. InnerCV protects outerfold choices, not accumulated adaptive research selection. Practical gate is not statistical significance or a prize guarantee. Public scores never tune settings.',
 'no_holdout_test_selection':True,'cost':0}
 write(_repo_path(OUT, 'plan.json'),p);check();emit({'prepared':True,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'gate':p['original_gate'],'regression':p['regression']})

def extract():
 check();ids,_=scope();start=time.perf_counter();sc=Scattering2D(J=6,shape=(256,256),L=8,max_order=2);X=[];hashes={};times=[]
 for i,ident in enumerate(ids):
  path=_repo_path(ROOT, 'data/train', f'{ident}.png');hashes[ident]=sha(path)
  with Image.open(_repo_io(path)) as im:raw=np.asarray(im)
  assert raw.shape==(256,256) and raw.dtype==np.uint8
  x=raw.astype(np.float32)/255.;t=time.perf_counter();a=sc(x);times.append(time.perf_counter()-t)
  assert a.shape==(1009,4,4) and np.isfinite(a).all()
  if i<3:
   sc.max_order=1;b=sc(x);sc.max_order=2;np.testing.assert_array_equal(a[:49],b)
  X.append(a.mean((-2,-1),dtype=np.float64))
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2e9 and time.perf_counter()-start<1800
  if (i+1)%50==0:emit({'extracted':i+1,'seconds':time.perf_counter()-start})
 X=np.stack(X);assert X.shape==(400,1009) and np.isfinite(X).all()
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X)
 r={'shape':list(X.shape),'dtype':str(X.dtype),'ids':ids,'image_sha256':hashes,'features_sha256':sha(_repo_path(OUT, 'development_features.npz')),'first3_order1_prefix_exact':True,'perimage_median_seconds':float(np.median(times)),'elapsed_seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'labels_used':False,'holdout_test_access':False};write(_repo_path(OUT, 'extraction.json'),r);emit({k:v for k,v in r.items() if k not in ('ids','image_sha256')})

def fit_model(X,y,alpha):
 mean=float(y.mean());std=float(y.std());assert std>0
 model=Pipeline([('scale',StandardScaler()),('regressor',KernelRidge(alpha=alpha,kernel='rbf',gamma=1./X.shape[1]))]);model.fit(X,(y-mean)/std)
 return {'model':model,'target_mean':mean,'target_std':std,'alpha':alpha,'dimension':X.shape[1]}
def predict(model,X):return model['model'].predict(X)*model['target_std']+model['target_mean']
def fit_arm(X,y,g,train,k):
 inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20270320+k).split(X[train],groups=g[train]));ip=np.full((3,len(train)),np.nan);audit=[]
 for j,(itr,iva) in enumerate(inner):
  assert not set(g[train[itr]])&set(g[train[iva]])
  audit.append({'inner_fold':j,'train_indices':train[itr].tolist(),'validation_indices':train[iva].tolist()})
  for ai,alpha in enumerate(ALPHAS):ip[ai,iva]=predict(fit_model(X[train[itr]],y[train[itr]],alpha),X[train[iva]])
 assert np.isfinite(ip).all();mse=np.mean((ip-y[train])**2,axis=1);best=int(np.argmin(mse));model=fit_model(X[train],y[train],ALPHAS[best]);model.update({'inner_folds':audit,'inner_oof':ip,'inner_mse':mse,'selected_index':best});return model

def compare(y,old,new,fid):
 gains=[rmse(y[fid==k],old[fid==k])-rmse(y[fid==k],new[fid==k]) for k in range(5)]
 return {'gain':rmse(y,old)-rmse(y,new),'fold_gains':gains,'positive_folds':sum(x>0 for x in gains)}
def run(split):
 p=check()
 if split!='original':assert read(_repo_path(OUT, 'original/report.json'))['advance']
 ids,y,fid,ref,g=inputs(split);ex=read(_repo_path(OUT, 'extraction.json'));assert sha(_repo_path(OUT, 'development_features.npz'))==ex['features_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].copy()
 dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);start=time.perf_counter();ps={'reference011':ref};modelhash={};records={}
 for arm,dim in DIMS.items():
  oof=np.full(400,np.nan);records[arm]=[]
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);m=fit_arm(X[:,:dim],y,g,tr,k);oof[va]=predict(m,X[va,:dim]);m.update({'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va],'train_indices':tr,'validation_indices':va,'arm':arm,'split':split});path=dest/f'{arm}-fold{k}.joblib';joblib.dump(m,_repo_io(path));modelhash[path.name]=sha(path);records[arm].append({'fold':k,'alpha':m['alpha'],'inner_mse':m['inner_mse'].tolist()});emit({'split':split,'arm':arm,'fold_complete':k,'alpha':m['alpha'],'seconds':time.perf_counter()-start})
  assert np.isfinite(oof).all();ps[arm]=oof;ps[arm+'_blend']=.9*ref+.1*oof
 comps={a:compare(y,ref,ps[a+'_blend'],fid) for a in DIMS};pair=compare(y,ps['order1'],ps['order2'],fid);pairblend=compare(y,ps['order1_blend'],ps['order2_blend'],fid);mask=~np.isin(ids,FIXED5);fixedgain=rmse(y[mask],ref[mask])-rmse(y[mask],ps['order2_blend'][mask]);assert mask.sum()==395
 advance=comps['order2']['gain']>=.02 and comps['order2']['positive_folds']>=3 and pair['gain']>0 and pairblend['gain']>0 and fixedgain>0
 with (dest/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),y=y,fold=fid,**ps)
 r={'split':split,'scores':{n:{'rmse':rmse(y,z),'fold_rmse':[rmse(y[fid==k],z[fid==k]) for k in range(5)]} for n,z in ps.items()},'comparisons_vs011':comps,'order2_vs_order1':pair,'order2blend_vs_order1blend':pairblend,'fixed395_gain_vs011':fixedgain,'advance':bool(advance),'inner_selection':records,'models_sha256':modelhash,'predictions_sha256':sha(dest/'development_predictions.npz'),'seconds':time.perf_counter()-start,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'candidate_created':False};write(dest/'report.json',r);check();emit({k:v for k,v in r.items() if k not in ('models_sha256','inner_selection')})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','extract',*SPLITS]);a=ap.parse_args();{'prepare':prepare,'extract':extract}.get(a.mode,lambda:run(a.mode))()
