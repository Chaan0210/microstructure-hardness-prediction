"""Fixed patch-origin averaging; replace only best011's short frozenB head."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np,torch
import dinob_joint_084 as e
import dinob_062 as b
import candidate006_pipeline as p6
OUT=_repo_path(ROOT, 'artifacts/patch-phase-104');SPLITS=('original','20261031','20261101');OFFSETS=((0,0),(0,7),(7,0),(7,7))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def emit(x):print(json.dumps(x,ensure_ascii=False),flush=True)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def scope():
 m=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));ids=sorted(m['development_folds']);assert len(ids)==400 and not set(ids)&set(m['holdout']);return ids,m

def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def prepare():
 OUT.mkdir(exist_ok=False);audit=read(_repo_path(ROOT, 'artifacts/pipeline-audit-103/report.json'));assert audit['passed'] and not audit['new_numerical_bug_found'];ids,_=scope()
 sources=['patch_phase_104.py','validate_patch_phase_104.py','dinob_joint_084.py','dinob_062.py','candidate006_pipeline.py'];inputs=['artifacts/pipeline-audit-103/report.json','artifacts/run-001/split_manifest.json','artifacts/dinob-062/development_features.npz','artifacts/dinob-062/extraction.json','reports/frozen-complement-085-validation.json']+[f'artifacts/frozen-complement-085/{s}/development_predictions.npz' for s in SPLITS]
 p={'study':'104 one fixed ViTB patch-grid phase averaging comparison','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'The arbitrary14px patch partition may introduce nuisance sensitivity tosmall imageoffsets. Averagingfour fixed origins may improve morphology representation without newbackbone orhead capacity. It also changes edgecontext andposition, so this isnot apurealiasing measurement orprovenlabelinvariance.',
 'duplicate_audit':'No pixelshift/patch-origin averaging implementation found.039 compares504fullframe with196crops/scales;046 averagesD4 rotations/reflections;042 translation testwasadditivetokenoffset,notspatialtranslation. Thosefailedfamilies are notrerun.',
 'features':'Apply unchanged full256->504 PILbicubic, grayscaleRGB, fixedImageNetaffine. Reflect-pad normalizedinput onbottom/right by7pixels, then504windows at(y,x) offsets(0,0),(0,7),(7,0),(7,7). No circularwrap,rotation,D4,contrast normalization orlearneddenoising. Extract frozenB normalizedblock6/9/12 CLS/patchmean/populationstd each6912. Mean fourviews infloat64, keepdimension6912. Includesidentityview exactly; eachshiftedview modifies atmost7/504 peraxis edgecontext. 7 ishalf14patchsize fixedfromarchitecture,notselectedfromlabels.',
 'offsets':[list(z) for z in OFFSETS],'control':'Verified011 samefold frozenhead/009 predictions reused; originalidentityfeatures must reproduce062 exactly. No needlesscontrolrefit; headarchitecture/init/order/schedule matchexisting011.',
 'config':e.CFG,'fit':'Unchanged084 frozen-arm fit, foldtrain-only StandardScaler andtargetmean/std. Head32 GELU dropout.5,12epochs,seed20270215+fold,effective16/micro2. Backbonefrozen, noinnerhyperparametersearch. Newfeatures areimage-local frozenand safeacrossfolds; alllearnedstatistics andhead useoutertrainonly.',
 'formula':'.9 * samefold009 + .1 * newphasehead; compare samefold011. No newweight orcalibration. Keep S/old B/legacy branchesunchanged.',
 'original_rule':'Evaluate complete original5fold. If pooledRMSE strictly improves011 andindependentverificationpasses, run BOTH existingresplits unchanged. Otherwise recordfailure andstop thisfixedtrial; nooffset/count/weight sweep.',
 'confirmation_rule':'Bothresplits must strictly improve011 pooledRMSE. Report foldgains andpairedconditionaluncertainty, but noaddedposthoc gates. Positiveall3 isempiricalrobustness onreused400,notindependentsignificance. Tinygains willbeidentified as such.',
 'resources':'Existinguv CPU2/MPS. Benchmarkthreeimages/fourviews before400 extraction; stoponlyif projectedextraction>1200seconds orRSSorMPSdriver>4GB. No newinstalls/cost.',
 'scope':'Development400only. No train.csv/holdoutlabels,testimages orpredictions usedforselection. Final500/testonlyaftercandidatefixed andallchecks; uploadreservedforparent.',
 'bias':'Historicallyreused400 andsameknownresplits; changinghypothesis addsadaptive selectionbias. No guaranteedimprovement.',
 'sources':{n:sha(_repo_path(ROOT, n)) for n in sources},'inputs':{n:sha(_repo_path(ROOT, n)) for n in inputs},'protected011':read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'))['protected_sha256'],'development_ids':ids,'cost':0}
 write(_repo_path(OUT, 'plan.json'),p);check();emit({'prepared':True,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'rule':p['original_rule']})

def views(a):
 pad=np.pad(a,((0,0),(0,7),(0,7)),mode='reflect');return [np.ascontiguousarray(pad[:,dy:dy+504,dx:dx+504]) for dy,dx in OFFSETS]

def extract():
 p=check();torch.set_num_threads(2);assert torch.backends.mps.is_available();ids,_=scope();model=b.model().to('mps');before=e.h.state_hash(model.state_dict());ex=read(_repo_path(ROOT, 'artifacts/dinob-062/extraction.json'))
 with np.load(_repo_path(ROOT, 'artifacts/dinob-062/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;old=z['X'].astype(float)
 arr=[];start=time.perf_counter();benchmark=None
 for i,ident in enumerate(ids):
  path=_repo_path(ROOT, 'data/train', f'{ident}.png');assert sha(path)==ex['image_sha256'][ident];a=p6.input_image(path);vv=views(a);np.testing.assert_array_equal(vv[0],a);values=[]
  for view in vv:values.append(b.feature(model,torch.from_numpy(view[None]).to('mps')).cpu().numpy()[0])
  z=np.stack(values);np.testing.assert_array_equal(z[0],old[i]);arr.append(z)
  if i==2:
   elapsed=time.perf_counter()-start;benchmark={'three_images_seconds':elapsed,'projected400_seconds':elapsed*400/3,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'three_images_reused':True};benchmark['passed']=benchmark['projected400_seconds']<1200 and benchmark['rss_peak_bytes']<4e9 and benchmark['mps_driver_bytes']<4e9;write(_repo_path(OUT, 'benchmark.json'),benchmark);emit(benchmark);assert benchmark['passed']
  if (i+1)%40==0:emit({'extracted':i+1,'seconds':time.perf_counter()-start})
 assert e.h.state_hash(model.state_dict())==before and all(q.grad is None for q in model.parameters());V=np.stack(arr);X=V.mean(1,dtype=np.float64);assert X.shape==(400,6912) and np.isfinite(X).all()
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X,views=V)
 report={'shape':list(X.shape),'views_shape':list(V.shape),'identity_all400_exact':True,'weights_unchanged':True,'features_sha256':sha(_repo_path(OUT, 'development_features.npz')),'seconds':time.perf_counter()-start,'image_sha256':ex['image_sha256'],'holdout_test_access':False};write(_repo_path(OUT, 'extraction.json'),report);check();emit({k:v for k,v in report.items() if k!='image_sha256'})

def inputs(split):
 check();ids,m=scope();v=read(_repo_path(ROOT, 'reports/frozen-complement-085-validation.json'));path=_repo_path(ROOT, f'artifacts/frozen-complement-085/{split}/development_predictions.npz');assert sha(path)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:assert z['ids'].tolist()==ids;d={k:z[k].copy() for k in ('y','fold','primary','reference009','frozen')}
 ex=read(_repo_path(OUT, 'extraction.json'));assert sha(_repo_path(OUT, 'development_features.npz'))==ex['features_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].copy()
 g=np.array([m['group_by_id'][i] for i in ids]);fid=d['fold'];sp=m if split=='original' else read(_repo_path(ROOT, f'artifacts/frozen-validation-021/seed-{split}/split_manifest.json'));np.testing.assert_array_equal(fid,[sp['development_folds'][i] for i in ids])
 for k in range(5):assert np.sum(fid==k)==80 and not set(g[fid==k])&set(g[fid!=k])
 np.testing.assert_array_equal(.9*d['reference009']+.1*d['frozen'],d['primary'])
 return {'ids':ids,'y':d['y'],'fid':fid,'g':g,'B':X,'reference009':d['reference009'],'reference011':d['primary']}

def comparison(y,a,z,fid):
 gain=rmse(y,a)-rmse(y,z);fg=[rmse(y[fid==k],a[fid==k])-rmse(y[fid==k],z[fid==k]) for k in range(5)];draw=np.random.default_rng(20271004).integers(0,len(y),(3000,len(y)));boot=np.sqrt(((y-a)[draw]**2).mean(1))-np.sqrt(((y-z)[draw]**2).mean(1));return {'gain':gain,'fold_gains':fg,'positive_folds':sum(t>0 for t in fg),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'CI_limit':'Conditional onobservedpredictions; ignoresadaptiveselectionandrefitting.'}

def run(split):
 check()
 if split!='original':assert read(_repo_path(OUT, 'original/report.json'))['advance'] and read(_repo_path(ROOT, 'reports/patch-phase-104-original-validation.json'))['passed']
 torch.set_num_threads(2);d=inputs(split);dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);model=b.model();start=time.perf_counter();pred=np.full(400,np.nan);records={}
 for k in range(5):
  pv,record=e.fit_arm(d,None,model,k,'frozen',dest/f'fold-{k}');pred[d['fid']==k]=pv;records[str(k)]=record;torch.mps.empty_cache()
 ps={'reference009':d['reference009'],'reference011':d['reference011'],'phase_head':pred,'primary':.9*d['reference009']+.1*pred};assert all(np.isfinite(z).all() for z in ps.values());comp=comparison(d['y'],ps['reference011'],ps['primary'],d['fid'])
 with (dest/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**ps)
 result={'split':split,'scores':{k:{'rmse':rmse(d['y'],z),'fold_rmse':[rmse(d['y'][d['fid']==i],z[d['fid']==i]) for i in range(5)]} for k,z in ps.items()},'comparison':comp,'advance':comp['gain']>0,'fold_report_sha256':{str(k):sha(dest/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':sha(dest/'development_predictions.npz'),'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'seconds':time.perf_counter()-start,'candidate_created':False};write(dest/'report.json',result);check();emit(result)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','extract',*SPLITS]);a=ap.parse_args();{'prepare':prepare,'extract':extract}.get(a.mode,lambda:run(a.mode))()
