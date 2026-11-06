"""Locked115 residual selector on verified full500 honest anchors; no new upstreamfit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,csv,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nonlinear_residual_115 as n
import validate_nonlinear_residual_115 as v
import candidate012_residual_pipeline as c
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
e=n.e;r=n.r;OUT=_repo_path(ROOT, 'artifacts/final-model-116');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-013.csv')

def prepare():
 n.check();c.check();decision=e.read(n.OUT/'decision.json');assert decision['eligible_for_candidate'];assert not CSV.exists();OUT.mkdir(exist_ok=False);validation=e.read(c.OUT/'training-validation.json');assert validation['passed'] and validation['fit_sha256']==e.sha(c.OUT/'fit.json')
 inputs={str(p.relative_to(ROOT)):e.sha(p) for p in [n.OUT/'decision.json',_repo_path(ROOT, 'reports/nonlinear-residual-115-summary.json'),c.OUT/'fit.json',c.OUT/'training-validation.json',c.OUT/'train57.npz',c.OUT/'test57.npz',_repo_path(ROOT, 'reports/candidate012-residual-verification.json')]}
 # Reuse inputs as immutable assets; test57 bytes hashed for integrity, not decoded untilfit complete.
 for j in range(3):
  path=c.OUT/f'block-{j}';br=e.read(path/'report.json');assert br==e.read(c.OUT/'fit.json')['blocks'][j];assert e.sha(path/'anchor.npz')==br['anchor_sha256'];inputs[str((path/'anchor.npz').relative_to(ROOT))]=br['anchor_sha256']
  for file in path.rglob('*'):
   if file.is_file():inputs[str(file.relative_to(ROOT))]=e.sha(file)
 p={'candidate':'013','source_study':'115','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision':decision,'formula':'unchangedcandidate011 + uniformmean3 selected residualcorrections','procedure':'Exactly115fivecandidate selector none/Ridge1/Ridge.1/RBF1/RBF.1 on3full500honest residualblocks166/167 from114. Sameinner3fold seed20271013+j;feature/kernel/targetstatistics trainonly;fullselectedfitwholeblock. No weight/gamma/epochsearch, no globalOOFresidual fit.','reuse':'Verified114 upstreammodels andnested011target caches unchanged;full500011 retained. Trainonlyresidualmodels;all500covered across3blocks.','sources':{x:e.sha(_repo_path(ROOT, x)) for x in ['candidate013_pipeline.py','nonlinear_residual_115.py','validate_nonlinear_residual_115.py']},'inputs':inputs,'protected':n.check()['protected'],'cost':0,'test':'Decodecachedtest57 onlyafterfullresidualfit andindependenttrainvalidation; inferenceandintegritychecks only.','uploaded':False};e.write(_repo_path(OUT, 'decision.json'),p);e.emit({'locked':True,'candidate':'013'})

def check():
 n.check();c.check();p=e.read(_repo_path(OUT, 'decision.json'))
 for x,h in {**p['sources'],**p['inputs'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, x))==h,x
 return p

def fit():
 check();d=c.data();X=c.getX();blocks=r.load('original')['blocks'];start=time.perf_counter();records=[];maxerr=0.;covered=[];count=0
 for j in range(3):
  source=c.OUT/f'block-{j}';path=_repo_path(OUT, f'block-{j}');path.mkdir(exist_ok=False)
  with np.load(_repo_io(source/'anchor.npz'),allow_pickle=False) as z:A=z['train_indices'];B=z['residual_indices'];res=d['y'][B]-z['p011']
  assert not set(A)&set(B) and set(A)|set(B)==set(range(500)) and not set(d['g'][A])&set(d['g'][B]);covered.extend(B.tolist());folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+j).split(X[B],res,d['g'][B]));ip=np.full((5,len(B)),np.nan);manual=np.full_like(ip,np.nan);models={}
  for t,(it,iv) in enumerate(folds):
   for ci,kind in enumerate(n.KINDS):
    m=n.fit(X[B[it]],res[it],blocks,kind);pp=n.predict(m,X[B[iv]]);ip[ci,iv]=pp;manual[ci,iv]=v.manual(m,X[B[it]],res[it],X[B[iv]],blocks);err=float(abs(manual[ci,iv]-pp).max());assert err<2e-9;maxerr=max(maxerr,err);file=path/f'inner-{t}-{kind}.joblib';joblib.dump({'model':m,'train_indices':B[it],'validation_indices':B[iv]},_repo_io(file));np.testing.assert_array_equal(n.predict(joblib.load(_repo_io(file))['model'],X[B[iv]]),pp);models[file.name]=e.sha(file);count+=1
    if kind in r.CANDIDATES:
     old=joblib.load(_repo_io(source/f'residual/inner-{t}-{kind}.joblib'));np.testing.assert_array_equal(old['train_indices'],B[it]);np.testing.assert_array_equal(r.predict(old['model'],X[B[iv]]),pp)
  scores=np.mean((ip-res[None,:])**2,1);mscores=np.mean((manual-res[None,:])**2,1);np.testing.assert_allclose(scores,mscores,atol=2e-9,rtol=0);chosen=n.KINDS[int(scores.argmin())];assert chosen==n.KINDS[int(mscores.argmin())];m=n.fit(X[B],res,blocks,chosen);pp=n.predict(m,X);ind=v.manual(m,X[B],res,X,blocks);np.testing.assert_allclose(pp,ind,atol=2e-9,rtol=0);joblib.dump({'model':m,'train_indices':B,'train_ids':[d['ids'][i] for i in B],'chosen':chosen},_repo_io(path/'selected.joblib'));np.testing.assert_array_equal(n.predict(joblib.load(_repo_io(path/'selected.joblib'))['model'],X),pp);count+=1
  with (path/'fit.npz').open('xb') as f:np.savez_compressed(_repo_io(f),block_indices=B,anchor_indices=A,residual=res,inner_predictions=ip,unscored_train_predictions=pp)
  rec={'block':j,'chosen':chosen,'n_residual_train':len(B),'n_anchor_train':len(A),'inner_MSE':dict(zip(n.KINDS,scores.tolist())),'inner_splits':[{'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()} for it,iv in folds],'models_sha256':models,'selected_sha256':e.sha(path/'selected.joblib'),'fit_sha256':e.sha(path/'fit.npz')};e.write(path/'report.json',rec);records.append(rec);e.emit({'block':j,'chosen':chosen})
 assert sorted(covered)==list(range(500));report={'n_train':500,'blocks':records,'train_ids':d['ids'],'independent_training_validation':{'passed':True,'models_reloaded':count,'manual_eigensolve_max_error':maxerr,'all_inner_choices_independently_recomputed':True,'linear_candidates_exact012':True,'anchor_exclusion_checked':True,'all500covered':True,'unscored_train_predictions_not_validation':True},'new_upstream_training':False,'seconds':time.perf_counter()-start,'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'no_test_decoding':True,'no_holdout_subgroup_score':True};e.write(_repo_path(OUT, 'fit.json'),report);check();e.emit({'fit_complete':True,'seconds':report['seconds']})

def outputs():
 check();assert e.read(_repo_path(OUT, 'fit.json'))['independent_training_validation']['passed'];ids,_=c.cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False)
 with np.load(_repo_io(c.OUT/'test57.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].copy()
 with (_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
 assert [v['ID'] for v in rows]==ids;old=np.array([float(v['hardness']) for v in rows]);cor=[]
 for j in range(3):
  path=_repo_path(OUT, f'block-{j}');assert e.sha(path/'selected.joblib')==e.read(path/'report.json')['selected_sha256'];cor.append(n.predict(joblib.load(_repo_io(path/'selected.joblib'))['model'],X))
 cor=np.stack(cor);p=old+cor.mean(0);assert np.isfinite(p).all();return ids,X,old,cor,p

def predict():
 ids,X,old,cor,p=outputs();assert len(ids)==len(set(ids))==1000
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,p))
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','incumbent011','correction0','correction1','correction2','candidate013']);w.writerows(zip(ids,old,*cor,p))
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),incumbent011=old,corrections=cor,primary=p)
 lo,hi=e.read(_repo_path(ROOT, 'artifacts/final-model-065/report.json'))['training_label_range'];m={'candidate':'013','study':'115','rows':1000,'columns':['ID','hardness'],'submission_sha256':e.sha(CSV),'components_sha256':e.sha(CSV.with_suffix('.components.csv')),'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'sample_submission_sha256':e.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'selected':[x['chosen'] for x in e.read(_repo_path(OUT, 'fit.json'))['blocks']],'prediction_range':[float(p.min()),float(p.max())],'training_label_range':[lo,hi],'outside_training_range':int(((p<lo)|(p>hi)).sum()),'correction_RMS':float(np.sqrt(np.mean(cor.mean(0)**2))),'clipping':False,'finite':True,'duplicates':0,'ID_order_exact':True,'test_fit_or_selection':False,'uploaded':False,'cost':0};e.write(CSV.with_suffix('.manifest.json'),m);e.emit({'CSV_ready':True,'SHA256':e.sha(CSV)})

def verify():
 ids,TX,old,cor,p=outputs();d=c.data();X=c.getX();blocks=r.load('original')['blocks'];manual=[]
 for j in range(3):
  path=_repo_path(OUT, f'block-{j}')
  with np.load(_repo_io(path/'fit.npz'),allow_pickle=False) as z:B=z['block_indices'];res=z['residual']
  m=joblib.load(_repo_io(path/'selected.joblib'))['model'];manual.append(v.manual(m,X[B],res,TX,blocks))
 manual=np.stack(manual);err=float(abs(manual-cor).max());assert err<2e-9;np.testing.assert_allclose(old+manual.mean(0),p,atol=2e-9,rtol=0)
 with CSV.open(newline='') as f:reader=csv.DictReader(f);assert reader.fieldnames==['ID','hardness'];rows=list(reader)
 assert [v['ID'] for v in rows]==ids;np.testing.assert_array_equal([float(v['hardness']) for v in rows],p)
 with np.load(_repo_path(OUT, 'predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['primary'],p);np.testing.assert_array_equal(z['corrections'],cor)
 with CSV.with_suffix('.reproduced.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,p))
 assert e.sha(CSV.with_suffix('.reproduced.csv'))==e.sha(CSV);man=e.read(CSV.with_suffix('.manifest.json'));assert e.sha(CSV)==man['submission_sha256'] and e.sha(CSV.with_suffix('.components.csv'))==man['components_sha256']
 with c.CSV.open(newline='') as f:prev=list(csv.DictReader(f))
 assert [v['ID'] for v in prev]==ids;previous=np.array([float(v['hardness']) for v in prev]);check();out={'valid':True,'candidate':'013','rows':1000,'ID_order_exact':True,'duplicates':0,'finite':True,'prediction_range':man['prediction_range'],'training_label_range':man['training_label_range'],'outside_training_range':man['outside_training_range'],'CSV_SHA256':e.sha(CSV),'CSV_byte_exact':True,'model_reload_exact':True,'independent_kernel_eigensolve_max_error':err,'selected':man['selected'],'test_fit_or_tuning':False,'holdout_subgroup_scored':False,'011_012_preserved':True,'uploaded':False,'cost':0,'difference_from012_RMS':float(np.sqrt(np.mean((p-previous)**2))),'training_validation':e.read(_repo_path(OUT, 'fit.json'))['independent_training_validation'],'source_sha256':e.sha(Path(__file__))};e.write(_repo_path(ROOT, 'reports/candidate013-verification.json'),out);e.emit(out)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','fit','predict','verify']);a=ap.parse_args();globals()[a.mode]()
