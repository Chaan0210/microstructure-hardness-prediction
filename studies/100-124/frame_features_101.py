"""One fixed TV all-region diameter extension, development400 only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,csv,time,hashlib,datetime,argparse,types,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np,joblib
from PIL import Image
import denoised_grain_features as dg
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/frame-features-101');SPLITS=('original','20261031','20261101');DIMS={'control57':57,'extension59':59}
FIXED5=('TRAIN_000438','TRAIN_000331','TRAIN_000227','TRAIN_000071','TRAIN_000168')
NEW_NAMES=['denoised_grain_median_all_area_weighted_diam','denoised_grain_range_all_area_weighted_diam']
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(x):print(json.dumps(x,ensure_ascii=False),flush=True)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def scope():
 m=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));ids=sorted(m['development_folds']);assert len(ids)==400 and not set(ids)&set(m['holdout']);return ids,m

def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def prepare():
 OUT.mkdir(exist_ok=False);ids,_=scope();old=read(_repo_path(ROOT, 'artifacts/run-006/report.json'));panel=read(_repo_path(ROOT, 'artifacts/run-022/report.json'))
 assert 'grain_s1_all_area_weighted_diam' in old['feature_names'] and 'grain_s2_all_area_weighted_diam' in old['feature_names']
 assert len(panel['features'])==57 and not any(x in panel['features'] for x in NEW_NAMES)
 src=['frame_features_101.py','validate_frame_features_101.py','grain_features.py','denoised_grain_features.py','compact_regression.py','frozen_validation.py']
 inp=['artifacts/run-006/report.json','artifacts/run-022/report.json','artifacts/run-022/development_features.npz','artifacts/run-022/development_oof.csv','artifacts/run-022/input-provenance.json','artifacts/run-001/split_manifest.json','reports/frame-censoring-100-summary.json','reports/frame-censoring-100-validation.json','reports/frozen-complement-085-validation.json','reports/candidate011-public-submission.json']+[f'artifacts/frozen-complement-085/{s}/development_predictions.npz' for s in SPLITS]
 p={'study':'101 single fixed TV visible-area diameter extension','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'duplicate_audit':{'identical':False,'run006':'Gaussian sigma1/2 -> closing blackhat/Otsu -> distance-transform h-maxima1.5 -> ridge watershed; all and interior area-weighted diameters kept separately per sigma. No TV, no four-configuration median/range.','proposed':'Existing canonicalD4 TV noise*.5/1 -> Gaussian gradient sigma.7 -> h-minima*.1/.2 q90 with existing dark-ridge exclusion; all area>=16 including frame; median/range over four configurations. Same geometric statistic but different fixed segmenter and aggregation.','novelty_limit':'Partial feature-family overlap; no claim of entirely new physical information.'},
 'arms':DIMS,'features':panel['features']+NEW_NAMES,'blocks':panel['blocks']+['denoised_grain_size']*2,
 'extraction':'Call unchanged extract_denoised bytecode with private globals and a summary observer. Observer computes area-weighted2sqrt(area/pi) over positive labels area>=16, including frame, with0 if none. Verify returned31 against old cache for ALL400. Append median and range of four values; preserve first57 exactly. No thresholds/markers/scales chosen using labels; native raw stays local.',
 'regression':'Identical existing BalancedKernel for both arms: foldtrain MinMaxScaler(clip=True), StandardScaler, inverse-sqrt block-size weights; RBF gamma=1/(median positive train squared distance * bandwidth^2); train target standardization. Existing bandwidth(.25,1,4) x alpha(.01,.1,1) selected by3-fold group CV seed20261005+outerfold, refitting all transformations within each innertrain. New columns belong to existing denoised_grain_size block; its inverse-sqrt column-count weights update by existing rule.',
 'blend':'.9 * same-fold verified011 + .1 * arm; only this weight; no replacement or fitted calibration.',
 'original_gate':{'minimum_extension_blend_gain_vs011':.02,'minimum_positive_folds_vs011':3,'extension_standalone_strictly_better_than_control':True,'extension_blend_strictly_better_than_control_blend':True,'fixed395_extension_blend_gain_vs011_strictly_positive':True},
 'resplit_gate':'Only if original gate passes and independent validation passes, evaluate BOTH existing seeds20261031/20261101 unchanged. Each must satisfy same entire original gate. No fallback weight, feature subset, threshold search or repeat. Allthree required before any consideration of full500 fit/test.',
 'fixed5_exclusions':list(FIXED5),'development_ids':ids,'sources':{n:sha(_repo_path(ROOT, n)) for n in src},'inputs':{n:sha(_repo_path(ROOT, n)) for n in inp},'protected011':read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'))['protected_sha256'],
 'bias':'Repeated adaptive use of the same400 including existing resplits. NestedCV controls this inner parameter selection only, not historical feature/research selection. Gains are not independent confirmation, significance, or a guarantee of ranking improvement.',
 'resource':'Existing uv .venv, CPU2, no installs, no network/data transmission, cost0. Extraction cap1800s/2GB. Only development raw400; holdout/test forbidden before final model decision.', 'cost':0}
 write(_repo_path(OUT, 'plan.json'),p);check();emit({'prepared':True,'duplicate_identical':False,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'original_gate':p['original_gate']})

def extract_one(image):
 vals=[]
 def observe(seg,x):
  a=np.bincount(seg.ravel())[1:].astype(float);a=a[a>=16]
  vals.append(float(np.sum(a*2*np.sqrt(a/np.pi))/a.sum()) if len(a) else 0.)
  return dg._summarize(seg,x)
 gl=dict(dg.extract_denoised.__globals__);gl['_summarize']=observe
 fn=types.FunctionType(dg.extract_denoised.__code__,gl)
 old,_,_=fn(image);assert len(vals)==4
 return old,np.array([np.median(vals),np.ptp(vals)]),np.array(vals)

def extract():
 check();ids,m=scope();start=time.perf_counter()
 with np.load(_repo_path(ROOT, 'artifacts/run-022/development_features.npz'),allow_pickle=False) as z:
  assert z['ids'].tolist()==ids;X=z['X'].copy();groups=z['groups'].copy()
 assert X.shape==(400,57);np.testing.assert_array_equal(groups,[m['group_by_id'][i] for i in ids])
 prior=read(_repo_path(ROOT, 'artifacts/run-022/input-provenance.json'))['new_image_sha256'];extra=[];rawvals=[];hashes={};err=0.
 for i,ident in enumerate(ids):
  path=_repo_path(ROOT, 'data/train', f'{ident}.png');hashes[ident]=sha(path);assert hashes[ident]==prior[ident]
  with Image.open(_repo_io(path)) as im:a=np.asarray(im,dtype=np.uint8).copy()
  assert a.shape==(256,256);old,add,raw=extract_one(a);np.testing.assert_allclose(old,X[i,-31:],rtol=0,atol=1e-12);err=max(err,float(abs(old-X[i,-31:]).max()));extra.append(add);rawvals.append(raw)
  assert time.perf_counter()-start<1800 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2e9
  if (i+1)%40==0:emit({'extracted':i+1,'seconds':time.perf_counter()-start,'old31_max_error':err})
 A=np.column_stack([X,np.stack(extra)]);assert A.shape==(400,59) and np.isfinite(A).all();np.testing.assert_array_equal(A[:,:57],X)
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),groups=groups,X=A,raw_four_config_diameters=np.stack(rawvals))
 r={'shape':list(A.shape),'old31_all400_max_error':err,'old57_exact':True,'image_sha256':hashes,'features_sha256':sha(_repo_path(OUT, 'development_features.npz')),'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'labels_used':False,'holdout_test_access':False};write(_repo_path(OUT, 'extraction.json'),r);emit({k:v for k,v in r.items() if k!='image_sha256'})

def inputs(split):
 check();ids,m=scope();v=read(_repo_path(ROOT, 'reports/frozen-complement-085-validation.json'));path=_repo_path(ROOT, f'artifacts/frozen-complement-085/{split}/development_predictions.npz');assert sha(path)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();fid=z['fold'].copy();ref=z['primary'].copy()
 g=np.array([m['group_by_id'][i] for i in ids])
 expected=m if split=='original' else read(_repo_path(ROOT, f'artifacts/frozen-validation-021/seed-{split}/split_manifest.json'))
 np.testing.assert_array_equal(fid,[expected['development_folds'][i] for i in ids])
 for k in range(5):assert np.sum(fid==k)==80 and not set(g[fid==k])&set(g[fid!=k])
 assert abs(rmse(y,ref)-v['results'][split]['scores']['primary']['rmse'])<1e-10
 return ids,y,fid,ref,g

def compare(y,old,new,fid):
 gains=[rmse(y[fid==k],old[fid==k])-rmse(y[fid==k],new[fid==k]) for k in range(5)]
 return {'gain':rmse(y,old)-rmse(y,new),'fold_gains':gains,'positive_folds':sum(x>0 for x in gains)}
def gate(y,ps,fid,ids):
 comp=compare(y,ps['reference011'],ps['extension59_blend'],fid);mask=~np.isin(ids,FIXED5);assert mask.sum()==395
 fixedgain=rmse(y[mask],ps['reference011'][mask])-rmse(y[mask],ps['extension59_blend'][mask]);pair=compare(y,ps['control57'],ps['extension59'],fid);bp=compare(y,ps['control57_blend'],ps['extension59_blend'],fid)
 checks={'gain_ge_002':comp['gain']>=.02,'positive_folds_ge3':comp['positive_folds']>=3,'standalone_beats57':pair['gain']>0,'blend_beats57blend':bp['gain']>0,'fixed395_positive':fixedgain>0}
 return {'advance':bool(all(checks.values())),'checks':checks,'extension_blend_vs011':comp,'extension_vs57':pair,'extensionblend_vs57blend':bp,'fixed395_gain_vs011':fixedgain}

def run(split):
 p=check()
 if split!='original':assert read(_repo_path(OUT, 'original/report.json'))['gate']['advance'] and read(_repo_path(ROOT, 'reports/frame-features-101-original-validation.json'))['passed']
 ids,y,fid,ref,g=inputs(split);ex=read(_repo_path(OUT, 'extraction.json'));assert sha(_repo_path(OUT, 'development_features.npz'))==ex['features_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].copy()
 dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);start=time.perf_counter();ps={'reference011':ref};mh={};logs={}
 for arm,dim in DIMS.items():
  pred=np.full(400,np.nan);logs[arm]=[]
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);pred[va],model,detail=fv.kernel_fold(X[:,:dim],y,g,p['blocks'][:dim],tr,va,k)
   saved={'model':model,'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va],'train_indices':tr,'validation_indices':va,'detail':detail,'arm':arm,'split':split};path=dest/f'{arm}-fold{k}.joblib';joblib.dump(saved,_repo_io(path));mh[path.name]=sha(path);logs[arm].append({'fold':k,**detail});emit({'split':split,'arm':arm,'fold_complete':k,'selected':detail['selected'],'seconds':time.perf_counter()-start})
  assert np.isfinite(pred).all();ps[arm]=pred;ps[arm+'_blend']=.9*ref+.1*pred
 replay=None
 if split=='original':
  with (_repo_path(ROOT, 'artifacts/run-022/development_oof.csv')).open() as f:old={r['ID']:r for r in csv.DictReader(f)}
  assert set(old)==set(ids);np.testing.assert_array_equal(y,[float(old[i]['hardness']) for i in ids]);prev=np.array([float(old[i]['combined_kernel']) for i in ids]);replay=float(abs(prev-ps['control57']).max());np.testing.assert_allclose(prev,ps['control57'],rtol=0,atol=1e-8)
 with (dest/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),y=y,fold=fid,**ps)
 r={'split':split,'scores':{a:{'rmse':rmse(y,z),'fold_rmse':[rmse(y[fid==k],z[fid==k]) for k in range(5)]} for a,z in ps.items()},'gate':gate(y,ps,fid,ids),'old57_control_prediction_max_error':replay,'inner_selection':logs,'models_sha256':mh,'predictions_sha256':sha(dest/'development_predictions.npz'),'seconds':time.perf_counter()-start,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'candidate_created':False};write(dest/'report.json',r);check();emit({k:v for k,v in r.items() if k not in ('inner_selection','models_sha256')})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','extract',*SPLITS]);a=ap.parse_args();{'prepare':prepare,'extract':extract}.get(a.mode,lambda:run(a.mode))()
