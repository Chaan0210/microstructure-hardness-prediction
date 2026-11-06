"""One final batch trial: finite-window third-point extension of matched pairs."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,time,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np,joblib
import amplitude_cnn_106 as base
import frozen_validation as fv
import threepoint_features_108 as f
OUT=_repo_path(ROOT, 'artifacts/threepoint-108');SPLITS=base.SPLITS;DIMS={'control57':57,'triple85':85};read=base.read;sha=base.sha;write=base.write;emit=base.emit;rmse=base.rmse

def check():
 p=read(_repo_path(OUT, 'plan.json'));base.timecheck()
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def prepare():
 assert not read(_repo_path(ROOT, 'artifacts/persistence-107/original/report.json'))['gate']['advance'];assert read(_repo_path(ROOT, 'reports/persistence-107-original-validation.json'))['passed'];OUT.mkdir(exist_ok=False);old=read(_repo_path(ROOT, 'artifacts/run-012/report.json'));assert len(old['features'])==len(old['blocks'])==26
 src=['threepoint_features_108.py','threepoint_study_108.py','validate_threepoint_108.py','frozen_validation.py','compact_regression.py','validate_frame_features_101.py'];inp=['artifacts/persistence-107/original/report.json','reports/persistence-107-original-validation.json','artifacts/run-012/development_features.npz','artifacts/run-012/report.json','artifacts/amplitude-cnn-106/plan.json','artifacts/amplitude-cnn-106/images.npy','artifacts/amplitude-cnn-106/image-cache.json','artifacts/run-001/split_manifest.json','reports/frozen-complement-085-validation.json']+[f'artifacts/frozen-complement-085/{s}/development_predictions.npz' for s in SPLITS]
 blocks=old['blocks']+['raw_intensity']*3+['pair_'+m for r in f.RADII for kind in ('line','corner') for m in ('mean','std')]+['triple_'+m for r in f.RADII for kind in ('line','corner') for m in ('mean','std')]
 p={'study':'108 fixed three-spatial-site central moment extension','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':base.DEADLINE,'development_ids':base.check()['development_ids'],'hypothesis':'A product at three distinct locations carries spatial intensity interactions not determined by two-point correlation or single-pixel skewness. It may complement current learned descriptors; utility unknown. Not a grain segmentation or denoising model.',
 'duplicate_audit':'Existing baseline and034autocov use two-point correlations;054 lineal uses binary contiguoussegment events;055 signed-distance Betti;107 grayscalecomponent pairing. Existing single-pixel intensitymoments do not equal nonzero-lagthree-site moments. No spatialtripleproduct/bispectrum feature found. Generictexture overlap acknowledged; no claim of physical causal identification.',
 'definition':'Raw256uint8 /255, subtract thisimage mean without variance/noise normalization. At r=1,2,4,8,16,32,64, two motifs: (0,0),(0,r),(0,2r) and (0,0),(0,r),(r,0). EightD4 orientations; each uses onlyvalid triplets completelyinsideframe, no wrap/padding/smoothing/threshold. For each orientation take mean centeredA*B*C. Preserve mean andpopulationSD across8orientations:28newfeatures. No Gaussian-noise independence assumption or claim of unbiasednoise correction.',
 'control':'Existingcompact26 plus rawmean/centralvariance/singlepixelcentral3, plus28 paired moments: on each identicaltripletsupport mean(AB+AC+BC)/3, thenD4mean/std. Control57 versus appendtriple28 giving85. Singlepointthirdmoment includedinboth. Raw intensity amplitudes notdividedbytinyvariance.',
 'blocks':blocks,'dimensions':DIMS,'training':'Unchanged existing BalancedKernel: foldtrainMinMaxScaler clip=True,StandardScaler,1/sqrt(blocksize),RBFmediantrain distance. NestedGroupKFold3shuffle seed20261005+k, widths.25/1/4 alpha.01/.1/1. Everytransform insideinnertrain/outertrain. Samepipelinebotharms. Addingblocks changeskernelgeometry; notmatchedparametercount.',
 'formula':'.9*samefold011+.1*arm. Onlytriple85 primary; control57 notalternatecandidate. No fittedweights,calibration or residualtraining.',
 'selection':'Originalprimary muststrictlyimprove011 andmatchedcontrolblend pooledRMSE. Ifpass plus independentvalidation thenBOTHknownresplits unchanged;all3mustpass beforefinalfit/test. No lag/motif/kernelgrid/blendsearch orfallback. Final planned hypothesis in thissmallbatch; completeconfirmation ifeligible within2h.',
 'engineering':'Vectorizedslice products checked by rotatedimage/indexcoordinateimplementation on toysand3realimages. D4 invariance,constant/contrastsign behavior. Mathematicaltoy uses equalcyclicpaircorrelation butdistincttriplecorrelation; onlydemonstratesnonredundancy, not productionperiodicboundary. First3realimages benchmark projected400<1200s,RSS<2GB; cache reused.',
 'limits':'Finitewindow, D4summary andlimitedmotifs loseexact topology andabsoluteorientation. Norealgroundtruth makesgrain-size fidelityunproven. Historicaldevelopmentreuse selectionbias remains; resplits reuse400.',
 'sources':{n:sha(_repo_path(ROOT, n)) for n in src},'inputs':{n:sha(_repo_path(ROOT, n)) for n in inp},'protected011':base.check()['protected011'],'scope':'ExistingvenvCPU2,cost0,noinstalls,norawexport,holdout/testunusedforselection.','cost':0};write(_repo_path(OUT, 'plan.json'),p);check();emit({'prepared':True,'plan_sha256':sha(_repo_path(OUT, 'plan.json'))})

def engineering():
 rng=np.random.default_rng(108);errs=[]
 for a in [np.full((16,16),128,dtype=np.uint8),rng.integers(0,256,(16,16),dtype=np.uint8)]:
  z=f.moments(a,(1,2,4));q=f.independent(a,(1,2,4))
  for x,y in zip(z,q):errs.append(float(abs(x-y).max()));np.testing.assert_allclose(x,y,rtol=0,atol=1e-15)
  inv=f.moments(255-a,(1,2,4));np.testing.assert_allclose(inv[1],z[1],rtol=0,atol=1e-15);np.testing.assert_allclose(inv[2][::2],-z[2][::2],rtol=0,atol=1e-15);np.testing.assert_allclose(inv[2][1::2],z[2][1::2],rtol=0,atol=1e-15)
 A=np.array([int(i in {0,1,4,6}) for i in range(12)],float);B=np.array([int(i in {0,1,3,7}) for i in range(12)],float);A-=A.mean();B-=B.mean();pa=np.array([np.dot(A,np.roll(A,k)) for k in range(12)]);pb=np.array([np.dot(B,np.roll(B,k)) for k in range(12)]);np.testing.assert_allclose(pa,pb,rtol=0,atol=1e-15);gap=abs(np.mean(A*np.roll(A,6)*np.roll(A,7))-np.mean(B*np.roll(B,6)*np.roll(B,7)));assert gap>.08
 return {'two_toy_independent_max_error':max(errs),'contrast_sign_identities':True,'cyclic_toy_all12pairlags_equal':True,'cyclic_toy_triple_gap':float(gap),'cyclic_toy_not_production_boundary':True}

def extract():
 p=check();eng=engineering();ids=p['development_ids'];images=np.load(_repo_io(base.OUT/'images.npy'),mmap_mode='r',allow_pickle=False)
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;old=z['X'].copy()
 assert old.shape==(400,26);start=time.perf_counter();values=[];maxerr=0.
 for i in range(400):
  base.timecheck();a=np.array(images[i,0]);parts=f.extract(a);row=np.concatenate(parts);assert row.shape==(59,);values.append(row)
  if i<3:
   independent=np.concatenate(f.independent(a));err=float(abs(row-independent).max());maxerr=max(maxerr,err);np.testing.assert_allclose(row,independent,rtol=0,atol=1e-15)
  if i==0:
   for k in range(4):
    for flip in (False,True):
     v=np.rot90(a,k);v=np.fliplr(v) if flip else v;np.testing.assert_allclose(row,np.concatenate(f.extract(v)),rtol=0,atol=1e-15)
  if i==2:
   sec=time.perf_counter()-start;r={'three_images_with_independent_QA_seconds':sec,'projected400_seconds':sec*400/3,'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'passed':sec*400/3<1200 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2e9};write(_repo_path(OUT, 'benchmark.json'),r);emit(r);assert r['passed']
  if (i+1)%40==0:emit({'extracted':i+1,'seconds':time.perf_counter()-start})
 X=np.column_stack([old,np.stack(values)]);assert X.shape==(400,85) and np.isfinite(X).all();np.testing.assert_array_equal(X[:,:26],old)
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),X=X)
 r={'shape':[400,85],'engineering':eng,'native3_independent_feature_max_error':maxerr,'native1_D4_exact_to_1e15':True,'old26_exact':True,'features_sha256':sha(_repo_path(OUT, 'development_features.npz')),'seconds':time.perf_counter()-start,'labels_used':False,'holdout_test_used':False};write(_repo_path(OUT, 'extraction.json'),r);check();emit(r)

def inputs(split):
 p=check();d=base.inputs(split);ex=read(_repo_path(OUT, 'extraction.json'));assert sha(_repo_path(OUT, 'development_features.npz'))==ex['features_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];d['features']=z['X'].copy()
 return d

def gate(d,ps):
 a=base.compare(d['y'],d['ref'],ps['triple85_blend'],d['fid']);b=base.compare(d['y'],ps['control57_blend'],ps['triple85_blend'],d['fid']);return {'advance':a['gain']>0 and b['gain']>0,'primary_vs011':a,'primary_vscontrolblend':b}

def run(split):
 p=check()
 if split!='original':assert read(_repo_path(OUT, 'original/report.json'))['gate']['advance'] and read(_repo_path(ROOT, 'reports/threepoint-108-original-validation.json'))['passed']
 d=inputs(split);dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);start=time.perf_counter();ps={'reference011':d['ref']};models={};selections={}
 for arm,dim in DIMS.items():
  X=d['features'][:,:dim];pred=np.full(400,np.nan)
  for k in range(5):
   base.timecheck();tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);pred[va],m,detail=fv.kernel_fold(X,d['y'],d['g'],p['blocks'][:dim],tr,va,k);path=dest/f'{arm}-fold{k}.joblib';joblib.dump({'model':m,'detail':detail,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va]},_repo_io(path));models[path.name]=sha(path);selections[path.name]=detail['selected'];emit({'arm':arm,'fold_complete':k,'seconds':time.perf_counter()-start})
  assert np.isfinite(pred).all();ps[arm]=pred;ps[arm+'_blend']=.9*d['ref']+.1*pred
 with (dest/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**ps)
 draw=np.random.default_rng(20271008).integers(0,400,(3000,400));boot=np.sqrt(np.mean((d['y']-d['ref'])[draw]**2,1))-np.sqrt(np.mean((d['y']-ps['triple85_blend'])[draw]**2,1));r={'split':split,'scores':{a:{'rmse':rmse(d['y'],z),'fold_rmse':[rmse(d['y'][d['fid']==k],z[d['fid']==k]) for k in range(5)]} for a,z in ps.items()},'gate':gate(d,ps),'conditional_gain_CI95':np.percentile(boot,[2.5,97.5]).tolist(),'models_sha256':models,'selected':selections,'predictions_sha256':sha(dest/'development_predictions.npz'),'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'seconds':time.perf_counter()-start,'candidate_created':False};write(dest/'report.json',r);check();emit(r)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','extract',*SPLITS]);a=ap.parse_args();{'prepare':prepare,'extract':extract}.get(a.mode,lambda:run(a.mode))()
