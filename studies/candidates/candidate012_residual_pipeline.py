"""User-authorized exploratory113 residual candidate; preserve011 and all prior decisions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,csv,datetime,argparse,importlib.metadata
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import residual_physics_113 as r
import nested_fusion_112 as e
import candidate009_pipeline as p9
import candidate_pipeline as cp
import denoised_grain_features as dg
import candidate_features as cf
import numpy as np,torch,joblib
from PIL import Image
from sklearn.model_selection import GroupKFold
OUT=_repo_path(ROOT, 'artifacts/final-model-114');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-012.csv')

def prepare():
 r.check();a=e.read(_repo_path(ROOT, 'reports/residual-physics-113-pooled-addendum.json'));assert a['MSE_gain']>0
 assert all(e.read(r.OUT/f'validation-{s}.json')['passed'] for s in e.SPLITS) and e.read(r.OUT/'leakage-validation.json')['passed'];assert not CSV.exists();OUT.mkdir(exist_ok=False)
 src=['candidate012_residual_pipeline.py','residual_physics_113.py','nested_fusion_112.py','nested_calibration_074.py','candidate009_pipeline.py','candidate_features.py','denoised_grain_features.py','validate_candidate012_residual.py']
 inputs=['reports/residual-physics-113-pooled-addendum.json','reports/residual-physics-113-summary.json','reports/candidate011-verification.json','artifacts/final-model-075/report.json','artifacts/final-model-075/training-inputs.npz','artifacts/final-model-065/report.json']
 d={'candidate':'012','study':'113','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'Parent explicitly directs pooled all3-partition untrimmed squared-error criterion; removes agent-imposed each-partition-improvement requirement. Earlier plan/decision preserved; this addendum records changed operational rule after results, not retroactive preregistration.','pooled_development':a,'formula':'unchanged full500 candidate011 + mean of3 honest-block residual regressors; no shrinkage,clip,new ratio,or globalOOF residual fit','full500_procedure':'GroupKFold3 seed20270110 on500. Each333/334 anchor excludes its166/167 residual block. Anchor011=.9*nested-affine008+.1*shortB. Reuse exact075 middle008 fits; newly train3 deep008 insideeachanchor, fitOLS onlyanchorOOF; trainshortB sameanchor. Within each residualblock use3fold seed20271013+j selectingnone/Ridge1/Ridge.1, allstatistics trainlocal. Refitselected onwholeblock. Averageall3 includingzero. Full500011 retained unchanged.','training_size_limit':'Residualmodels use166/167 at production versus106/107 in development. Anchors333/334 versus213/214; unchangedprocedure, no assertion of size invariance.','seed_rule':'Final outer index0;middleoffset100*(j+1);deepseed20270110+offset;shortB base+offset;residualCV20271013+j,matchingdevelopment outer0 rule.','sources':{n:e.sha(_repo_path(ROOT, n)) for n in src},'inputs':{n:e.sha(_repo_path(ROOT, n)) for n in inputs},'protected011':r.check()['protected011'],'environment':r.check()['environment'],'test':'Only after fit lock and validation; raw features and savedmodel inference, no fit/tuning/selection.','holdout':'Included only as authorized full500 training; no formerholdout subgroup score.','resource':'Existing CPU2/MPS,9deep008 fits+3shortB,3small residualmodels, existing feature caches and image-only fixed57 extraction. No installs/cost/upload.','uploaded':False}
 e.write(_repo_path(OUT, 'decision.json'),d);e.emit({'locked':True,'pooled_gain':a['pooled_RMSE_gain'],'candidate':'012'})

def check():
 r.check();d=e.read(_repo_path(OUT, 'decision.json'))
 for n,h in {**d['sources'],**d['inputs'],**d['protected011']}.items():assert e.sha(_repo_path(ROOT, n))==h,n
 assert {v.metadata['Name'].lower().replace('_','-'):v.version for v in importlib.metadata.distributions()}==d['environment'];return d

def data():
 check();d=p9.load_training();ids,y=cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);assert d['ids']==ids;np.testing.assert_array_equal(d['y'],y)
 sp=e.read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));np.testing.assert_array_equal(d['g'],[sp['group_by_id'][i] for i in ids]);return d

def one(ident,split,compact):
 path=_repo_path(ROOT, 'data', split, f'{ident}.png')
 with Image.open(_repo_io(path)) as im:assert im.mode=='L' and im.size==(256,256);a=np.asarray(im).copy()
 tv,n,b=dg.extract_denoised(a)
 if compact:
  K,kn=cf.extract_candidate(a,'run-012');x=np.r_[K,tv];names=kn+n
 else:x=tv;names=n
 assert np.isfinite(x).all();return x,names,e.sha(path)

def train_features():
 check();d=data();old=r.load('original');lookup={v:i for i,v in enumerate(old['ids'])};base=e.read(_repo_path(ROOT, 'artifacts/final-model-065/report.json'));X=np.empty((500,57));start=time.perf_counter();missing=[]
 for i,ident in enumerate(d['ids']):
  assert e.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==base['training_image_sha256'][ident]
  if ident in lookup:np.testing.assert_array_equal(d['K'][i],old['X'][lookup[ident],:26]);X[i]=old['X'][lookup[ident]]
  else:missing.append(i)
 for first in range(0,len(missing),20):
  ix=missing[first:first+20];rows=joblib.Parallel(n_jobs=2)(joblib.delayed(one)(d['ids'][i],'train',False) for i in ix)
  for i,(tv,n,h) in zip(ix,rows):assert n==e.read(_repo_path(ROOT, 'artifacts/run-022/report.json'))['features'][26:] and h==base['training_image_sha256'][d['ids'][i]];X[i]=np.r_[d['K'][i],tv]
  e.emit({'train57_added':first+len(ix),'seconds':time.perf_counter()-start})
 with (_repo_path(OUT, 'train57.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),X=X)
 e.write(_repo_path(OUT, 'train57.json'),{'sha256':e.sha(_repo_path(OUT, 'train57.npz')),'cached400_exact':True,'new100_image_only_features':True,'train_image_hashes_verified':500,'seconds':time.perf_counter()-start})

def getX():
 assert e.sha(_repo_path(OUT, 'train57.npz'))==e.read(_repo_path(OUT, 'train57.json'))['sha256']
 with np.load(_repo_path(OUT, 'train57.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==data()['ids'];return z['X'].copy()

def fit():
 check();torch.set_num_threads(2);d=data();X=getX();start=time.perf_counter();splits=list(GroupKFold(3,shuffle=True,random_state=e.cal.INNER_SEED).split(d['S'],d['y'],d['g']));records=[];source=e.read(p9.OUT/'report.json')
 for j,(A,B) in enumerate(splits):
  path=_repo_path(OUT, f'block-{j}');path.mkdir(exist_ok=False);offset=100*(j+1);middle=p9.OUT/f'inner-{j}';mr=e.read(middle/'report.json');assert e.sha(middle/'report.json')==source['inner_report_sha256'][f'inner-{j}/report.json'];assert mr['train_indices']==A.tolist() and mr['validation_indices']==B.tolist()
  for n,h in mr['files_sha256'].items():assert e.sha(middle/n)==h
  with np.load(_repo_io(middle/'predictions.npz'),allow_pickle=False) as z:p008=z['ensemble008'].copy()
  inner=list(GroupKFold(3,shuffle=True,random_state=e.cal.INNER_SEED+offset).split(d['S'][A],d['y'][A],d['g'][A]));nested=np.full(len(A),np.nan)
  for ell,(it,iv) in enumerate(inner):
   nested[iv],rec=e.cal.fit_cell(d,A[it],A[iv],B,offset,ell,path/f'deep-{ell}');e.emit({'block':j,'deep':ell,'elapsed':time.perf_counter()-start})
  center=float(nested.mean());ym=float(d['y'][A].mean());z=nested-center;slope=float(z@(d['y'][A]-ym)/(z@z));p009=ym+slope*(p008-center);short,srec=e.fit(d,None,None,A,B,0,j,'shortB',path/'shortB');p011=.9*p009+.1*short
  with (path/'anchor.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=A,residual_indices=B,inner_oof008=nested,p008=p008,p009=p009,shortB=short,p011=p011)
  rd={'ids':d['ids'],'X':X,'y':d['y'],'g':d['g'],'blocks':r.load('original')['blocks'],'fid':np.zeros(500,dtype=int)};_,rr=r.fit_block(rd,B,p011,0,j,path/'residual')
  br={'block':j,'anchor_rows':len(A),'residual_rows':len(B),'anchor_sha256':e.sha(path/'anchor.npz'),'middle_report_sha256':e.sha(middle/'report.json'),'deep_report_sha256':{f'deep-{ell}/report.json':e.sha(path/f'deep-{ell}/report.json') for ell in range(3)},'shortB_report_sha256':e.sha(path/'shortB/report.json'),'residual_report_sha256':e.sha(path/'residual/report.json'),'selected':rr['chosen'],'calibrator':{'center':center,'target_mean':ym,'slope':slope}};e.write(path/'report.json',br);records.append(br);e.emit({'block_complete':j,'selected':rr['chosen'],'elapsed':time.perf_counter()-start})
 e.write(_repo_path(OUT, 'fit.json'),{'n_train':500,'train_ids':d['ids'],'blocks':records,'seconds':time.perf_counter()-start,'train_csv_sha256':e.sha(_repo_path(ROOT, 'data/train.csv')),'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'no_test_access':True,'no_holdout_subgroup_score':True});check();e.emit({'full500_fit_complete':True,'seconds':time.perf_counter()-start})

def test_features():
 check();assert e.read(_repo_path(OUT, 'training-validation.json'))['passed'];ids,_=cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);base=e.read(_repo_path(ROOT, 'artifacts/submission-candidate-008.manifest.json'));rows=[];start=time.perf_counter();schema=e.read(_repo_path(ROOT, 'artifacts/run-022/report.json'))['features']
 for first in range(0,1000,40):
  batch=joblib.Parallel(n_jobs=2)(joblib.delayed(one)(ident,'test',True) for ident in ids[first:first+40]);assert all(n==schema and h==base['test_image_sha256'][ident] for ident,(x,n,h) in zip(ids[first:first+40],batch));rows.extend(batch);e.emit({'test57_extracted':len(rows),'seconds':time.perf_counter()-start})
 X=np.stack([v[0] for v in rows]);assert X.shape==(1000,57) and np.isfinite(X).all()
 with (_repo_path(OUT, 'test57.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X)
 e.write(_repo_path(OUT, 'test57.json'),{'sha256':e.sha(_repo_path(OUT, 'test57.npz')),'image_sha256':{ident:v[2] for ident,v in zip(ids,rows)},'seconds':time.perf_counter()-start,'sample_submission_sha256':e.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'fit_frozen_sha256':e.sha(_repo_path(OUT, 'fit.json'))})

def outputs():
 check();ids,_=cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);assert e.sha(_repo_path(OUT, 'test57.npz'))==e.read(_repo_path(OUT, 'test57.json'))['sha256']
 with np.load(_repo_path(OUT, 'test57.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].copy()
 with (_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
 assert [v['ID'] for v in rows]==ids;old=np.array([float(v['hardness']) for v in rows]);cor=[]
 for j in range(3):
  path=_repo_path(OUT, f'block-{j}/residual');rec=e.read(path/'report.json');assert e.sha(path/'selected.joblib')==rec['selected_sha256'];m=joblib.load(_repo_io(path/'selected.joblib'))['model'];cor.append(r.predict(m,X))
 cor=np.stack(cor);pred=old+cor.mean(0);assert np.isfinite(pred).all();return ids,old,cor,pred

def predict():
 ids,old,cor,pred=outputs();assert len(ids)==len(set(ids))==1000
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','incumbent011','correction0','correction1','correction2','candidate012']);w.writerows(zip(ids,old,*cor,pred))
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),incumbent011=old,corrections=cor,primary=pred)
 lo,hi=e.read(_repo_path(ROOT, 'artifacts/final-model-065/report.json'))['training_label_range'];e.write(CSV.with_suffix('.manifest.json'),{'candidate':'012','study':'113','rows':1000,'columns':['ID','hardness'],'submission_sha256':e.sha(CSV),'components_sha256':e.sha(CSV.with_suffix('.components.csv')),'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'sample_submission_sha256':e.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'test57_sha256':e.sha(_repo_path(OUT, 'test57.npz')),'selected':[v['selected'] for v in e.read(_repo_path(OUT, 'fit.json'))['blocks']],'prediction_range':[float(pred.min()),float(pred.max())],'training_label_range':[lo,hi],'outside_training_range':int(((pred<lo)|(pred>hi)).sum()),'correction_RMS':float(np.sqrt(np.mean(cor.mean(0)**2))),'correction_range':[float(cor.mean(0).min()),float(cor.mean(0).max())],'finite':True,'exact_sample_order':True,'duplicates':0,'clipping':False,'test_fit_or_selection':False,'holdout_scored':False,'uploaded':False,'cost':0});e.emit({'CSV_ready':str(CSV),'sha256':e.sha(CSV)})
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train_features','fit','test_features','predict']);a=p.parse_args();globals()[a.mode]()
