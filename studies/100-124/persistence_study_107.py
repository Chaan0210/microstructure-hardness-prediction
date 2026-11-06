"""Fixed grayscale persistence image versus same-dimensional Betti curve."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import persistence_features_107 as f
import amplitude_cnn_106 as prev
OUT=_repo_path(ROOT, 'artifacts/persistence-107');ARMS=('curve','persistence');SPLITS=prev.SPLITS;ALPHAS=(51.2,512.,5120.)
read=prev.read;sha=prev.sha;write=prev.write;emit=prev.emit;rmse=prev.rmse
class Regressor:
 def __init__(self,alpha):self.alpha=alpha
 def fit(self,X,y):
  self.scaler=StandardScaler().fit(X);self.ym=float(y.mean());self.ys=float(y.std());self.ridge=Ridge(alpha=self.alpha,solver='cholesky').fit(self.scaler.transform(X),(y-self.ym)/self.ys);return self
 def predict(self,X):return self.ridge.predict(self.scaler.transform(X))*self.ys+self.ym

Regressor.__module__='persistence_study_107'
if __name__=='__main__':sys.modules['persistence_study_107']=sys.modules[__name__]

def check():
 p=read(_repo_path(OUT, 'plan.json'));prev.timecheck()
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def prepare():
 assert read(prev.OUT/'original/report.json')['advance'] is False
 assert read(_repo_path(ROOT, 'reports/amplitude-cnn-106-original-validation.json'))['passed']
 OUT.mkdir(exist_ok=False);ids=prev.check()['development_ids'];src=['persistence_features_107.py','persistence_study_107.py','validate_persistence_107.py'];inp=['artifacts/amplitude-cnn-106/original/report.json','reports/amplitude-cnn-106-original-validation.json','artifacts/amplitude-cnn-106/plan.json','artifacts/amplitude-cnn-106/images.npy','artifacts/amplitude-cnn-106/image-cache.json','artifacts/run-001/split_manifest.json','reports/frozen-complement-085-validation.json']+[f'artifacts/frozen-complement-085/{s}/development_predictions.npz' for s in SPLITS]
 p={'study':'107 fixed grayscale0D persistence image versus Betti curve','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':prev.DEADLINE,'development_ids':ids,'hypothesis':'Joint birth/death pairing of connected intensity excursions may add phase/interface connectivity information lost by threshold counts. Persistence weights attenuate short-lived fluctuations but do not establish noise removal or actual grain recovery.',
 'duplicate_audit':'055 computes signed-distance Betti curves at3 grayscale quantiles and explicitly is not persistence.054lineal,028connectedmask,TVwatershed and099slicedpatchWasserstein do not compute graylevel component birth/death pairing. No max-tree/persistence implementation found in existing research sources. New representation, not new denoising threshold or CNN resolution.',
 'definition':'Native raw256 uint8, no smoothing/resize/contrast normalization. Lower-level4-neighbor connectivity on raw and255-raw independently, finite rectangle no wrap/outside. Installed skimage max_tree supplies component hierarchy; reverse-tree elder rule yields positive-length finite bars. Omit the sole essential bar consistently. Endpoints integer0..255.',
 'control':'For each polarity,256 thresholds0..255: count finite bars with birth<=t<death, divide65536. Two curves512 features.',
 'candidate':'For each polarity map birth/255,lifetime=(death-birth)/255. Weight by lifetime; integrate isotropic Gaussian sigma1/16 into fixed16x16 cells covering[0,1]^2 using normal CDF. Do not renormalize Gaussian mass outsideunit square; divide65536. Two256 images512 features. Same dimension, but different basis/weighting as well as pairing; not a pure pairing-only ablation.',
 'references':['https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.max_tree','https://jmlr.org/papers/v18/16-337.html'],'reference_use':'Official component-treeAPI and persistence-image formulation. Own fixed application, no externaltrainingdata/codedownload, no claimed paperbenchmark reproduction or microstructure-accuracy theorem.',
 'training':'Both512 arms same StandardScaler and standardizedtarget Ridge. alpha51.2/512/5120 chosen via pooled3-fold group innerCV,shuffle seed20271007+outerfold. Fit transformations anew inside everyinnertrain and outertrain. No PCA/residualtargets/featurechoice. Primary=.9 samefold011+.1 persistence; curveblend matchedcontrol, not alternatecandidate.',
 'original_rule':'Primary persistenceblend must strictly improve011 and curveblend pooledRMSE. Ifpass and independentvalidationpasses, run BOTH existingresplits unchanged; samecriteriaall3 forpromotion. No grid/bin/sigma/polarity/weight search. Report folds and conditionalCI without interpreting as independent confirmation.',
 'engineering':'Independent pixel union-find on43 small images includinguniform/ties; all observedthresholdcounts versusndimage. Three development images all256 levels/polarities checked directly; one full256image union-find; allD4 of oneimage. Samecurve/differentpairing fixture. Three image runtime projection <1200s andRSS<2GB. Cache alreadyauthorized400 inputbytes withhashcheck.',
 'limits':'0D graph topology, not1Dholes/fullcubicalhomology; no pixelmetric size or spatialarrangement recovery. Boundarycensoring andunidentifiednoise remain. Same400 repeatedly used adaptively; innerCV doesnotremovehistoricalselectionbias. Syntheticexamples engineeringonly.',
 'sources':{n:sha(_repo_path(ROOT, n)) for n in src},'inputs':{n:sha(_repo_path(ROOT, n)) for n in inp},'protected011':prev.check()['protected011'],'scope':'Noholdout/test/extra labels/newinstalls/payments/rawexport. CPU2 existingvenv, 2h batchdeadline. Preserve011.','cost':0};write(_repo_path(OUT, 'plan.json'),p);check();emit({'prepared':True,'plan_sha256':sha(_repo_path(OUT, 'plan.json'))})

def extract():
 p=check();import validate_persistence_107 as v
 from scipy import ndimage as ndi
 start=time.perf_counter();eng=v.engineering();ids=p['development_ids'];X=np.load(_repo_io(prev.OUT/'images.npy'),mmap_mode='r',allow_pickle=False);assert X.shape==(400,1,256,256);curves=[];pis=[];benchstart=time.perf_counter()
 for i in range(400):
  prev.timecheck();a=np.array(X[i,0]);c,pi,pds=f.extract(a);curves.append(c);pis.append(pi)
  if i<3:
   for pol,(image,pd) in enumerate(zip((a,255-a),pds)):
    for t in range(256):assert ndi.label(image<=t)[1]==int(((pd[:,0]<=t)&(t<pd[:,1])).sum())+int(t>=image.min())
  if i==0:
   for image,pd in zip((a,255-a),pds):np.testing.assert_array_equal(pd,v.independent_diagram(image))
   for k in range(4):
    for flip in (False,True):
     z=np.rot90(a,k);z=np.fliplr(z) if flip else z;cc,pp,_=f.extract(np.ascontiguousarray(z));np.testing.assert_array_equal(cc,c);np.testing.assert_allclose(pp,pi,rtol=0,atol=1e-18)
  if i==2:
   # Conservative projection includes expensive one-off independentQA.
   elapsed=time.perf_counter()-benchstart;bench={'three_images_including_independent_QA_seconds':elapsed,'projected400_seconds':elapsed*400/3,'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'passed':elapsed*400/3<1200 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2e9};write(_repo_path(OUT, 'benchmark.json'),bench);emit(bench);assert bench['passed']
  if (i+1)%40==0:emit({'extracted':i+1,'seconds':time.perf_counter()-start})
 C=np.stack(curves);P=np.stack(pis);assert C.shape==P.shape==(400,512) and np.isfinite(C).all() and np.isfinite(P).all()
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),curve=C,persistence=P)
 report={'shape_per_arm':[400,512],'engineering':eng,'native3_all256levels_bothpolarities_exact':True,'native1_bothpolarities_unionfind_exact':True,'native1_D4_exact_to_1e18':True,'features_sha256':sha(_repo_path(OUT, 'development_features.npz')),'seconds':time.perf_counter()-start,'labels_used':False,'holdout_test_used':False};write(_repo_path(OUT, 'extraction.json'),report);check();emit(report)

def inputs(split):
 p=check();d=prev.inputs(split);ex=read(_repo_path(OUT, 'extraction.json'));assert sha(_repo_path(OUT, 'development_features.npz'))==ex['features_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];d.update({a:z[a].copy() for a in ARMS})
 return d

def nested(X,d,tr,va,k):
 y=d['y'];g=d['g'];splits=list(GroupKFold(n_splits=3,shuffle=True,random_state=20271007+k).split(X[tr],y[tr],g[tr]));scores=[]
 for alpha in ALPHAS:
  q=np.full(len(tr),np.nan)
  for it,iv in splits:
   assert not set(g[tr[it]])&set(g[tr[iv]]);m=Regressor(alpha).fit(X[tr[it]],y[tr[it]]);q[iv]=m.predict(X[tr[iv]])
  scores.append({'alpha':alpha,'rmse':rmse(y[tr],q)})
 chosen=min(scores,key=lambda z:z['rmse']);m=Regressor(chosen['alpha']).fit(X[tr],y[tr]);return m.predict(X[va]),m,{'inner_scores':scores,'selected':chosen,'inner_indices':[[tr[a].tolist(),tr[b].tolist()] for a,b in splits]}

def gate(d,ps):
 comp=prev.compare(d['y'],d['ref'],ps['persistence_blend'],d['fid']);paired=prev.compare(d['y'],ps['curve_blend'],ps['persistence_blend'],d['fid']);return {'advance':comp['gain']>0 and paired['gain']>0,'primary_vs011':comp,'primary_vscurveblend':paired}

def run(split):
 check()
 if split!='original':assert read(_repo_path(OUT, 'original/report.json'))['gate']['advance'] and read(_repo_path(ROOT, 'reports/persistence-107-original-validation.json'))['passed']
 d=inputs(split);dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);start=time.perf_counter();ps={'reference011':d['ref']};models={};details={}
 for arm in ARMS:
  X=d[arm];pred=np.full(400,np.nan)
  for k in range(5):
   prev.timecheck();tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);pred[va],m,detail=nested(X,d,tr,va,k);path=dest/f'{arm}-fold{k}.joblib';joblib.dump({'model':m,'detail':detail,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va]},_repo_io(path));models[path.name]=sha(path);details[path.name]=detail['selected'];emit({'arm':arm,'fold_complete':k,'seconds':time.perf_counter()-start})
  assert np.isfinite(pred).all();ps[arm]=pred;ps[arm+'_blend']=.9*d['ref']+.1*pred
 with (dest/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**ps)
 rng=np.random.default_rng(20271007);draw=rng.integers(0,400,(3000,400));boot=np.sqrt(np.mean((d['y']-d['ref'])[draw]**2,1))-np.sqrt(np.mean((d['y']-ps['persistence_blend'])[draw]**2,1))
 r={'split':split,'scores':{a:{'rmse':rmse(d['y'],z),'fold_rmse':[rmse(d['y'][d['fid']==k],z[d['fid']==k]) for k in range(5)]} for a,z in ps.items()},'gate':gate(d,ps),'conditional_gain_CI95':np.percentile(boot,[2.5,97.5]).tolist(),'models_sha256':models,'selected':details,'predictions_sha256':sha(dest/'development_predictions.npz'),'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'seconds':time.perf_counter()-start,'candidate_created':False};write(dest/'report.json',r);check();emit(r)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','extract',*SPLITS]);a=ap.parse_args();{'prepare':prepare,'extract':extract}.get(a.mode,lambda:run(a.mode))()
