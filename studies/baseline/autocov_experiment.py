"""Frozen autocovariance shape hypothesis; development400 only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,csv,hashlib,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from autocov_shape_features import extract_autocov,linear_covariance,SCALES,RADII,MAX_LAG
OUT=_repo_path(ROOT, 'artifacts/run-033')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,r):
 with p.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
def manifest():
 sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());ids=sorted(sp['development_folds']);assert len(ids)==400;return sp,ids
def read_image(i,ids):
 assert i in ids
 with Image.open(_repo_path(ROOT, 'data/train', f'{i}.png')) as im:assert im.mode=='L' and im.size==(256,256);return np.asarray(im,dtype=np.uint8).copy()
def rows(run,ids):
 with (_repo_path(ROOT, f'artifacts/{run}/development_oof.csv')).open() as f:r=list(csv.DictReader(f))
 b={z['ID']:z for z in r};assert len(r)==len(b)==400 and set(b)==set(ids);return b

def qa():
 OUT.mkdir(exist_ok=False)
 rng=np.random.default_rng(20261111);a=rng.integers(64,96,size=(256,256),dtype=np.uint8);y,x=np.indices((256,256));stripes=np.rint(128+40*np.sin(2*np.pi*x/32)).astype(np.uint8);blob=np.rint(180-80*np.exp(-((x-128)**2+(y-128)**2)/128)).astype(np.uint8)
 fixtures={'constant':np.full((256,256),128,np.uint8),'random':a,'stripes':stripes,'blob':blob};d4=0;direct_errors=[]
 for name,im in fixtures.items():
  f,n,b=extract_autocov(im);assert len(n)==len(set(n))==len(b)==36
  if name=='constant':np.testing.assert_array_equal(f,np.zeros(36))
  for k in range(4):
   for z in (np.rot90(im,k),np.fliplr(np.rot90(im,k))):np.testing.assert_allclose(extract_autocov(z)[0],f,rtol=1e-7,atol=1e-10);d4+=1
  centered=im.astype(float)/255;centered-=centered.mean()
  for scale in SCALES:
   sm=ndi.gaussian_filter(centered,scale,mode='reflect');cov=linear_covariance(sm)
   for dy,dx in ((0,0),(1,3),(2,0),(32,-5),(64,64)):
    v0=sm[max(0,-dy):min(256,256-dy),max(0,-dx):min(256,256-dx)];v1=sm[max(0,dy):min(256,256+dy),max(0,dx):min(256,256+dx)];direct=float(np.mean(v0*v1));error=abs(direct-cov[MAX_LAG+dy,MAX_LAG+dx]);assert error<1e-12;direct_errors.append(error)
 f=extract_autocov(a)[0];twice=extract_autocov(2*a)[0];np.testing.assert_allclose(f[:32],twice[:32],rtol=1e-7,atol=1e-10);np.testing.assert_allclose(twice[32:34],4*f[32:34],rtol=1e-7,atol=1e-10);np.testing.assert_allclose(twice[34],2*f[34],rtol=1e-7);np.testing.assert_allclose(twice[35],f[35],rtol=1e-7)
 r={'source_sha256':sha(_repo_path(ROOT, 'autocov_shape_features.py')),'D4_checks_passed':d4,'direct_covariance_checks':len(direct_errors),'direct_max_error':max(direct_errors),'contrast_scaling_shape_invariance':True,'synthetic_model_training_rows':0,'target_labels_read':False,'proceed_to_cv':True,'limits':'Unit fixtures verify numerical and invariance behavior, not physical grain truth. Gaussian filtering is a fixed measurement branch, not a mislabeled training augmentation.'}
 write(_repo_path(OUT, 'qa-engineering.json'),r);write(_repo_path(OUT, 'qa-review.json'),r);print(json.dumps(r),flush=True)
def fit():
 import frozen_validation as fv
 sp,ids=manifest();review=json.loads((_repo_path(OUT, 'qa-review.json')).read_text());assert review['source_sha256']==sha(_repo_path(ROOT, 'autocov_shape_features.py')) and review['proceed_to_cv']
 prior=rows('run-012',ids);cnnrows=rows('run-009',ids);y=np.array([float(prior[i]['hardness']) for i in ids]);np.testing.assert_array_equal(y,[float(cnnrows[i]['hardness']) for i in ids]);base=np.array([float(prior[i]['baseline']) for i in ids]);cnn=np.array([float(cnnrows[i]['cnn']) for i in ids]);oldkernel=np.array([float(prior[i]['nested_kernel']) for i in ids]);old=np.array([float(prior[i]['nested_kernel_cnnblend']) for i in ids]);np.testing.assert_allclose(old,.5*base+.25*cnn+.25*oldkernel,rtol=0,atol=1e-10)
 cache=_repo_path(ROOT, 'artifacts/run-012/development_features.npz')
 with np.load(_repo_io(cache),allow_pickle=False) as z:
  zi=list(z['ids']);assert len(zi)==len(set(zi))==400 and set(zi)==set(ids);oldX=z['X'][[zi.index(i) for i in ids]].copy()
 oldrep=json.loads((_repo_path(ROOT, 'artifacts/run-012/report.json')).read_text());groups=np.array([sp['group_by_id'][i] for i in ids]);fid=np.array([sp['development_folds'][i] for i in ids]);start=time.perf_counter()
 plan={'hypothesis':'All-direction nonzero-lag covariance shapes after fixed sigma1/2 smoothing may reveal length and alignment without brittle segmentation.','novelty':'Baseline519 has only axial7-lag autocorrelation normalized by raw variance; native28 has coarse3-band spectral harmonics/LoGtile energies. New descriptors sample angular covariance at8radii, use nonzero-lag RMS, separate amplitude/noise/blur, and avoid circular wrap.','scales':list(SCALES),'radii':list(RADII),'shape_normalization':'RMS covariance on nonzero radius2 ring; no noise variance subtraction','models':['fixed compact26+autocov36 nested KRR'],'primary_ensemble':{'baseline':.5,'original_single_CNN':.25,'new_kernel':.25},'inner_grid':{'width':list(fv.WIDTHS),'alpha':list(fv.ALPHAS),'folds':3},'source_sha256':{n:sha(_repo_path(ROOT, n)) for n in ('autocov_shape_features.py','autocov_experiment.py','compact_regression.py','frozen_validation.py')},'protected':{n:sha(_repo_path(ROOT, n)) for n in ('artifacts/submission-candidate-003.csv','artifacts/final-model-017/report.json')},'input_sha256':{n:sha(_repo_path(ROOT, n)) for n in ('artifacts/run-001/split_manifest.json','artifacts/run-012/development_features.npz','artifacts/run-012/development_oof.csv','artifacts/run-009/development_oof.csv','artifacts/run-033/qa-engineering.json','artifacts/run-033/qa-review.json')},'scope':'Development400 only. No synthetic training, external data, holdout/test reads, manual target correction or Public-score tuning.'}
 write(_repo_path(OUT, 'plan.json'),plan)
 vec=[];hashes={}
 for j,i in enumerate(ids):
  v,nn,nb=extract_autocov(read_image(i,ids));vec.append(v);hashes[i]=sha(_repo_path(ROOT, 'data/train', f'{i}.png'))
  if (j+1)%50==0:print(json.dumps({'extracted':j+1,'seconds':time.perf_counter()-start}),flush=True)
 N=np.stack(vec);X=np.column_stack([oldX,N]);names=oldrep['features']+nn;blocks=oldrep['blocks']+nb;assert X.shape==(400,62) and np.isfinite(X).all()
 np.savez_compressed(_repo_path(OUT, 'development_features.npz'),X=X,ids=np.array(ids),groups=groups);write(_repo_path(OUT, 'input-provenance.json'),{'image_sha256':hashes,'feature_cache_sha256':sha(_repo_path(OUT, 'development_features.npz'))});feature_seconds=time.perf_counter()-start
 pred=np.full(400,np.nan);control=np.full(400,np.nan);logs=[]
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);control[va],_,_=fv.kernel_fold(oldX,y,groups,oldrep['blocks'],tr,va,k);np.testing.assert_allclose(control[va],oldkernel[va],rtol=0,atol=1e-7)
  pred[va],_,details=fv.kernel_fold(X,y,groups,blocks,tr,va,k);logs.append({'fold':k,**details});print(json.dumps({'fold':k,'blend_rmse':fv.rmse(y[va],.5*base[va]+.25*cnn[va]+.25*pred[va])}),flush=True)
 new=.5*base+.25*cnn+.25*pred;vals={'candidate003':old,'old_kernel_control':control,'acf_kernel':pred,'acf_blend':new};scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in vals.items()}
 with (_repo_path(OUT, 'development_oof.csv')).open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness',*vals]);w.writerows([i,y[j],*[v[j] for v in vals.values()]] for j,i in enumerate(ids))
 rng=np.random.default_rng(20261108);idx=rng.integers(0,400,size=(2000,400));gain=np.sqrt(np.mean((y-old)[idx]**2,axis=1))-np.sqrt(np.mean((y-new)[idx]**2,axis=1))
 for n,h in plan['source_sha256'].items():assert sha(_repo_path(ROOT, n))==h
 for n,h in plan['protected'].items():assert sha(_repo_path(ROOT, n))==h
 report={**plan,'features':names,'blocks':blocks,'scores':scores,'folds':logs,'feature_seconds':feature_seconds,'elapsed_seconds':time.perf_counter()-start,'feature_cache_sha256':sha(_repo_path(OUT, 'development_features.npz')),'control_max_abs_difference':float(np.max(abs(control-oldkernel))),'conditional_improvement_ci95':list(map(float,np.percentile(gain,[2.5,97.5]))),'residual_corr_vs003':float(np.corrcoef(y-old,y-new)[0,1]),'limits':'Autocovariance mixes grain geometry, phase contrast, blur and noise; it is not true grain size. Historical400 selection bias remains. Synthetic fixtures are unit tests only.'};write(_repo_path(OUT, 'report.json'),report);print(json.dumps({'scores':scores,'elapsed_seconds':report['elapsed_seconds'],'conditional_improvement_ci95':report['conditional_improvement_ci95']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['qa','fit']);args=p.parse_args();qa() if args.mode=='qa' else fit()
