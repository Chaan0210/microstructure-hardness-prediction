"""Target-blind QA then one fixed nested-CV pore-curvature hypothesis."""
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
from PIL import Image,ImageDraw
from pore_curvature_features import extract_pore_curvature,maps,SCALES,THRESHOLD
OUT=_repo_path(ROOT, 'artifacts/run-030')
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
 OUT.mkdir(exist_ok=True);sp,ids=manifest();sel=json.loads((_repo_path(ROOT, 'artifacts/run-028/qa-selection.json')).read_text());print('qa-selection keys',list(sel),flush=True)
 # Reuse fixed label-blind noise/size strata IDs, never label/error selection.
 chosen=[]
 def collect(value):
  if isinstance(value,str) and value in ids and value not in chosen:chosen.append(value)
  elif isinstance(value,dict):
   for k,v in value.items():collect(k);collect(v)
  elif isinstance(value,list):
   for v in value:collect(v)
 collect(sel);assert len(chosen)==16,chosen
 yy,xx=np.indices((256,256));base=np.full((256,256),.8);pore=.35*np.exp(-((xx-128)**2+(yy-128)**2)/(2*3.**2));ridge=.2*np.exp(-(yy-128)**2/(2*1.5**2))
 fixtures={'constant':base,'isolated_pore':base-pore,'ridge_only':base-ridge,'attached_pore':base-pore-ridge}
 outcomes={}
 for name,a in fixtures.items():
  im=np.rint(255*a).astype(np.uint8);v,n,b=extract_pore_curvature(im);_,records=maps(im);response=[float(r['response'][128,128]) for r in records];outcomes[name]={'center_response':response,'features':dict(zip(n,map(float,v)))}
  for k in range(4):
   for t in (np.rot90(im,k),np.fliplr(np.rot90(im,k))):np.testing.assert_allclose(extract_pore_curvature(t)[0],v,rtol=1e-8,atol=1e-10)
 np.testing.assert_allclose(list(outcomes['constant']['features'].values()),0.,atol=1e-12)
 isolated=max(outcomes['isolated_pore']['center_response']);attached=max(outcomes['attached_pore']['center_response']);line=max(outcomes['ridge_only']['center_response']);assert isolated>THRESHOLD and attached>.75*isolated and line<.1*isolated
 for group in range(4):
  canvas=Image.new('RGB',(4*256,2*282),'white');draw=ImageDraw.Draw(canvas)
  for col,i in enumerate(chosen[group*4:(group+1)*4]):
   a=read_image(i,ids);_,rs=maps(a);raw=Image.fromarray(a).convert('RGB');overlay=raw.copy();d=ImageDraw.Draw(overlay)
   for r in rs:
    py,px=np.nonzero(r['peaks']);color={1.:'red',2.:'yellow',4.:'cyan',8.:'lime'}[r['scale']]
    for y,x in zip(py,px):d.ellipse((int(x)-2,int(y)-2,int(x)+2,int(y)+2),outline=color)
   canvas.paste(raw,(col*256,26));canvas.paste(overlay,(col*256,308));draw.text((col*256+2,3),i,fill='black');draw.text((col*256+2,286),'s1 red s2 yellow s4 cyan s8 green',fill='black')
  canvas.save(_repo_path(OUT, f'qa-pore-{group+1}.png'))
 write(_repo_path(OUT, 'qa-engineering.json'),{'source_sha256':sha(_repo_path(ROOT, 'pore_curvature_features.py')),'selected_ids':chosen,'selection_source_sha256':sha(_repo_path(ROOT, 'artifacts/run-028/qa-selection.json')),'target_labels_read':False,'fixtures_used_for_model_training':False,'fixtures':outcomes,'D4_checks_passed':32,'attached_over_isolated_peak_response':attached/isolated,'ridge_over_isolated_peak_response':line/isolated})
 print('QA fixtures passed;16 target-blind image overlays saved',flush=True)

def fit():
 import frozen_validation as fv
 sp,ids=manifest();review=json.loads((_repo_path(OUT, 'qa-review.json')).read_text());assert review['source_sha256']==sha(_repo_path(ROOT, 'pore_curvature_features.py')) and review['proceed_to_cv']
 prior=rows('run-012',ids);cnnrows=rows('run-009',ids);y=np.array([float(prior[i]['hardness']) for i in ids]);np.testing.assert_array_equal(y,[float(cnnrows[i]['hardness']) for i in ids]);base=np.array([float(prior[i]['baseline']) for i in ids]);cnn=np.array([float(cnnrows[i]['cnn']) for i in ids]);oldkernel=np.array([float(prior[i]['nested_kernel']) for i in ids]);old=np.array([float(prior[i]['nested_kernel_cnnblend']) for i in ids]);np.testing.assert_allclose(old,.5*base+.25*cnn+.25*oldkernel,rtol=0,atol=1e-10)
 cache=_repo_path(ROOT, 'artifacts/run-012/development_features.npz')
 with np.load(_repo_io(cache),allow_pickle=False) as z:
  zi=list(z['ids']);assert len(zi)==len(set(zi))==400 and set(zi)==set(ids);oldX=z['X'][[zi.index(i) for i in ids]].copy()
 oldrep=json.loads((_repo_path(ROOT, 'artifacts/run-012/report.json')).read_text());groups=np.array([sp['group_by_id'][i] for i in ids]);fid=np.array([sp['development_folds'][i] for i in ids]);start=time.perf_counter()
 plan={'hypothesis':'Positive minimum Hessian curvature can capture dark pores attached to long boundaries, unlike whole-component shape filtering.','novelty':'Native28 squared LoG tile energies lack minimum-eigenvalue local curvature and blob extrema. This is not watershed, closed-network counting or noise variance subtraction.','scales':list(SCALES),'absolute_response_threshold_floor':THRESHOLD,'peak_threshold':'max(.01,3*imageHHnoise*discrete normalized-dxx-kernel L2 norm); no noise variance subtraction','models':['fixed compact26+porecurvature22 nested KRR'],'primary_ensemble':{'baseline':.5,'original_single_CNN':.25,'new_kernel':.25},'inner_grid':{'width':list(fv.WIDTHS),'alpha':list(fv.ALPHAS),'folds':3},'source_sha256':{n:sha(_repo_path(ROOT, n)) for n in ('pore_curvature_features.py','pore_curvature_experiment.py','compact_regression.py','frozen_validation.py')},'protected':{n:sha(_repo_path(ROOT, n)) for n in ('artifacts/submission-candidate-003.csv','artifacts/final-model-017/report.json')},'input_sha256':{n:sha(_repo_path(ROOT, n)) for n in ('artifacts/run-001/split_manifest.json','artifacts/run-012/development_features.npz','artifacts/run-012/development_oof.csv','artifacts/run-009/development_oof.csv','artifacts/run-030/qa-engineering.json','artifacts/run-030/qa-review.json')},'scope':'Development400 only. No synthetic training, external data, holdout/test reads, manual target correction or Public-score tuning.'}
 write(_repo_path(OUT, 'plan.json'),plan)
 vec=[];hashes={}
 for j,i in enumerate(ids):
  v,nn,nb=extract_pore_curvature(read_image(i,ids));vec.append(v);hashes[i]=sha(_repo_path(ROOT, 'data/train', f'{i}.png'))
  if (j+1)%50==0:print(json.dumps({'extracted':j+1,'seconds':time.perf_counter()-start}),flush=True)
 N=np.stack(vec);X=np.column_stack([oldX,N]);names=oldrep['features']+nn;blocks=oldrep['blocks']+nb;assert X.shape==(400,48) and np.isfinite(X).all()
 np.savez_compressed(_repo_path(OUT, 'development_features.npz'),X=X,ids=np.array(ids),groups=groups);write(_repo_path(OUT, 'input-provenance.json'),{'image_sha256':hashes,'feature_cache_sha256':sha(_repo_path(OUT, 'development_features.npz'))});feature_seconds=time.perf_counter()-start
 pred=np.full(400,np.nan);control=np.full(400,np.nan);logs=[]
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);control[va],_,_=fv.kernel_fold(oldX,y,groups,oldrep['blocks'],tr,va,k);np.testing.assert_allclose(control[va],oldkernel[va],rtol=0,atol=1e-7)
  pred[va],_,details=fv.kernel_fold(X,y,groups,blocks,tr,va,k);logs.append({'fold':k,**details});print(json.dumps({'fold':k,'blend_rmse':fv.rmse(y[va],.5*base[va]+.25*cnn[va]+.25*pred[va])}),flush=True)
 new=.5*base+.25*cnn+.25*pred;vals={'candidate003':old,'old_kernel_control':control,'pore_kernel':pred,'pore_blend':new};scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in vals.items()}
 with (_repo_path(OUT, 'development_oof.csv')).open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness',*vals]);w.writerows([i,y[j],*[v[j] for v in vals.values()]] for j,i in enumerate(ids))
 rng=np.random.default_rng(20261108);idx=rng.integers(0,400,size=(2000,400));gain=np.sqrt(np.mean((y-old)[idx]**2,axis=1))-np.sqrt(np.mean((y-new)[idx]**2,axis=1))
 for n,h in plan['source_sha256'].items():assert sha(_repo_path(ROOT, n))==h
 for n,h in plan['protected'].items():assert sha(_repo_path(ROOT, n))==h
 report={**plan,'features':names,'blocks':blocks,'scores':scores,'folds':logs,'feature_seconds':feature_seconds,'elapsed_seconds':time.perf_counter()-start,'feature_cache_sha256':sha(_repo_path(OUT, 'development_features.npz')),'control_max_abs_difference':float(np.max(abs(control-oldkernel))),'conditional_improvement_ci95':list(map(float,np.percentile(gain,[2.5,97.5]))),'residual_corr_vs003':float(np.corrcoef(y-old,y-new)[0,1]),'limits':'No ground truth pore labels; Hessian also responds to dark phase, junctions and noise. Fixed small panel and reused400 still allow selection optimism. Synthetic QA only.'};write(_repo_path(OUT, 'report.json'),report);print(json.dumps({'scores':scores,'elapsed_seconds':report['elapsed_seconds'],'conditional_improvement_ci95':report['conditional_improvement_ci95']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['qa','fit']);a=p.parse_args();qa() if a.mode=='qa' else fit()
