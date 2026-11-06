"""Locked144 bootstrap model-selection aggregation of best017; no new representation."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,csv,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate017_pipeline as p
import bootstrap_selector_144 as s
import numpy as np,torch,joblib
e=s.e;a=s.a;n=s.n;OUT=_repo_path(ROOT, 'artifacts/final-model-145');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-018.csv')

def prepare():
 p.check();s.check();d=e.read(s.OUT/'decision.json');assert d['advance'] and d['validation_passed'];OUT.mkdir(exist_ok=False);assert not CSV.exists();names=['candidate018_pipeline.py','bootstrap_selector_144.py','artifacts/bootstrap-selector-144/decision.json','reports/bootstrap-selector-144-summary.json','reports/candidate017-public-submission.json','artifacts/final-model-141/fit.json','artifacts/final-model-141/training-validation.json']
 names += [f'artifacts/bootstrap-selector-144/validation-{k}.json' for k in e.SPLITS]
 for j in range(3):names += [f'artifacts/final-model-141/block-{j}/residual/{f}' for f in ['report.json','predictions.npz','selected.joblib']]
 r={'candidate':'018','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'development':d,'recipe':'Exactly017inputs/affine/base/3honestblocks; for eachblock512groupbootstrapdraws ofexisting5candidateinnerOOFloss determineconvexwinnerfrequencyweights. Fitall5candidatesonB,weightedmeanthenmean3blocks.','full500_seed':'20271204+block, no newseed/drawcount/model/weighttuning','reuse':'All017full500base/nestedmodels,affines,spatial585features,720-styleinnerselectorpredictions and3selectedresidualmodels; only9additionalnonnullBresidualfits. Noimages/backbone/neuralfit.','holdout':'500fitonlyafterfixeddevelopmentdecision;noformersubgroupscore/selection','test':'Onlyinferenceaftertrainingverification;reuseimmutable017features. Nofit/clipping/tuningon test;existingtrain-derivedfeatureMinMaxclip remains.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':s.check()['protected'],'best017_retained':True,'cost':0,'uploaded':False};e.write(_repo_path(OUT, 'decision.json'),r);e.emit({'candidate018_locked':True,'new_nonnull_fits':9,'new_neural_fits':0})

def check():
 p.check();s.check();r=e.read(_repo_path(OUT, 'decision.json'))
 for f,h in {**r['hashes'],**r['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 return r

def fit():
 check();d,P,cells=p.data();start=time.perf_counter();records=[]
 for j,(A,B,M,D,short) in enumerate(cells):
  path=_repo_path(OUT, f'block-{j}');path.mkdir();src=p.OUT/f'block-{j}/residual';r=e.read(src/'report.json');z=a.readnp(src/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(z['block_indices'],B);loss=(z['inner_predictions']-z['residual'])**2;w,draw,wins=s.weights(loss,d['g'][B],s.SEED+j);pred=[];hashes={};new=0
  for kind in s.KINDS:
   if kind==r['chosen']:
    obj=joblib.load(_repo_io(src/'selected.joblib'));np.testing.assert_array_equal(obj['train_indices'],B);m=obj['model']
   else:m=n.fit(d['X'][B],z['residual'],d['blocks'],kind);new+=kind!='none'
   f=path/f'{kind}.joblib';joblib.dump({'model':m,'train_indices':B},_repo_io(f));hashes[f.name]=e.sha(f);pred.append(n.predict(m,d['X']))
  pred=np.stack(pred);np.testing.assert_array_equal(pred[s.KINDS.index(r['chosen'])],z['correction'])
  with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),block_indices=B,weights=w,draws=draw,winners=wins,candidate_predictions=pred,correction=w@pred)
  rec={'block':j,'n_train':len(B),'seed':s.SEED+j,'weights':dict(zip(s.KINDS,w.tolist())),'old_hardwinner':r['chosen'],'new_nonnull_fits':int(new),'upstream_report_sha256':e.sha(src/'report.json'),'models_sha256':hashes,'predictions_sha256':e.sha(path/'predictions.npz')};e.write(path/'report.json',rec);records.append(rec);e.emit({'full500_block':j,'weights':rec['weights']})
 r={'rows':500,'train_ids':d['ids'],'new_nonnull_fits':sum(v['new_nonnull_fits'] for v in records),'new_neural_fits':0,'unchanged_017_anchor_and_innerfit_reused':True,'blocks':records,'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'seconds':time.perf_counter()-start,'holdout_subgroup_scored':False,'test_read':False};e.write(_repo_path(OUT, 'fit.json'),r);check();e.emit({'full500_complete':True,'seconds':r['seconds'],'new_nonnull_fits':r['new_nonnull_fits']})

def training():
 check();d,P,cells=p.data();fit=e.read(_repo_path(OUT, 'fit.json'));assert fit['train_ids']==d['ids'];maxerr=0.;count=0;covered=[]
 for j,(A,B,M,D,short) in enumerate(cells):
  src=p.OUT/f'block-{j}/residual';old=e.read(src/'report.json');oz=a.readnp(src/'predictions.npz',old['predictions_sha256']);np.testing.assert_array_equal(oz['block_indices'],B);np.testing.assert_allclose(oz['anchor_predictions'],.9*s.q.independent_affine(a.mixture(D,p.p.ARM),d['y'][A],a.mixture(M,p.p.ARM))+.1*short,atol=1e-9,rtol=0);res=d['y'][B]-oz['anchor_predictions'];np.testing.assert_array_equal(res,oz['residual']);loss=(oz['inner_predictions']-res)**2;path=_repo_path(OUT, f'block-{j}');r=e.read(path/'report.json');assert r==fit['blocks'][j];assert r['upstream_report_sha256']==e.sha(src/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);w,wins=s.independent_weights(loss,d['g'][B],z['draws']);np.testing.assert_array_equal(w,z['weights']);np.testing.assert_array_equal(wins,z['winners']);np.testing.assert_array_equal(w,list(r['weights'].values()));_,draw,_=s.weights(loss,d['g'][B],s.SEED+j);np.testing.assert_array_equal(draw,z['draws']);pp=[];covered.extend(B.tolist());assert not set(A)&set(B)
  for ci,kind in enumerate(s.KINDS):
   f=path/f'{kind}.joblib';assert e.sha(f)==r['models_sha256'][f.name];obj=joblib.load(_repo_io(f));np.testing.assert_array_equal(obj['train_indices'],B);assert obj['model']['kind']==kind;pred=n.predict(obj['model'],d['X']);manual=s.q.vn.manual(obj['model'],d['X'][B],res,d['X'],d['blocks']);err=float(abs(pred-manual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(pred,z['candidate_predictions'][ci]);pp.append(pred);count+=1
  np.testing.assert_allclose(sum(wi*pi for wi,pi in zip(w,pp)),z['correction'],atol=1e-12,rtol=0)
 assert sorted(covered)==list(range(500));r={'passed':True,'rows':500,'models_verified':count,'independent_max_error':maxerr,'bootstrap_frequency_weights_exact':True,'all500_honest_block_coverage':True,'017_training_certificate':e.sha(p.OUT/'training-validation.json'),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'test_read':False,'holdout_subgroup_scored':False};e.write(_repo_path(OUT, 'training-validation.json'),r);check();e.emit(r)

def inference(device='mps'):
 check();v=e.read(_repo_path(OUT, 'training-validation.json'));assert v['passed'] and v['fit_sha256']==e.sha(_repo_path(OUT, 'fit.json'));ids,raw,short,base,old,oldpred,X=p.inference(device);cor=[];allpp=[]
 for j in range(3):
  root=_repo_path(OUT, f'block-{j}');r=e.read(root/'report.json');w=np.array([r['weights'][kind] for kind in s.KINDS]);pp=[]
  for kind in s.KINDS:
   f=root/f'{kind}.joblib';assert e.sha(f)==r['models_sha256'][f.name];pp.append(n.predict(joblib.load(_repo_io(f))['model'],X))
  pp=np.stack(pp);allpp.append(pp);cor.append(w@pp)
 cor=np.stack(cor);pred=base+cor.mean(0);assert pred.shape==(1000,) and np.isfinite(pred).all();return ids,raw,short,base,np.stack(allpp),cor,pred,X,oldpred

def predict():
 ids,raw,short,base,allpp,cor,pred,X,oldpred=inference();assert len(ids)==len(set(ids))==1000
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),components=raw,short=short,base=base,candidate_predictions=allpp,corrections=cor,primary=pred,reference017=oldpred)
 r={'candidate':'018','rows':1000,'submission_sha256':e.sha(CSV),'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'training_validation_sha256':e.sha(_repo_path(OUT, 'training-validation.json')),'sample_submission_sha256':e.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'ID_order_exact':True,'finite':True,'duplicates':0,'prediction_range':[float(pred.min()),float(pred.max())],'difference_from017_RMS':float(np.sqrt(np.mean((pred-oldpred)**2))),'output_clipping':False,'feature_clipping':'Preexistingtrain-fittedMinMaxclip only','holdout_subgroup_scored':False,'test_fit_or_tuning':False,'uploaded':False,'cost':0};assert r['submission_sha256']!=e.sha(p.CSV);e.write(CSV.with_suffix('.manifest.json'),r);e.emit(r)

def final():
 ids,raw,short,base,allpp,cor,pred,X,oldpred=inference();m=e.read(CSV.with_suffix('.manifest.json'));assert e.sha(CSV)==m['submission_sha256'];z=a.readnp(_repo_path(OUT, 'predictions.npz'),m['predictions_sha256']);assert z['ids'].tolist()==ids
 for key,arr in [('components',raw),('short',short),('base',base),('candidate_predictions',allpp),('corrections',cor),('primary',pred),('reference017',oldpred)]:np.testing.assert_array_equal(arr,z[key])
 ids2,_,_,_,_,_,cpu,_,_=inference('cpu');assert ids2==ids;cpuerr=float(abs(cpu-pred).max());assert cpuerr<1e-3;d,P,cells=p.data();mc=[];maxerr=0.
 for j,(A,B,M,D,sh) in enumerate(cells):
  old=e.read(p.OUT/f'block-{j}/residual/report.json');oz=a.readnp(p.OUT/f'block-{j}/residual/predictions.npz',old['predictions_sha256']);res=oz['residual'];r=e.read(_repo_path(OUT, f'block-{j}/report.json'));pp=[]
  for ci,kind in enumerate(s.KINDS):
   model=joblib.load(_repo_path(OUT, f'block-{j}/{kind}.joblib'))['model'];manual=s.q.vn.manual(model,d['X'][B],res,X,d['blocks']);err=float(abs(manual-allpp[j,ci]).max());assert err<2e-9;maxerr=max(maxerr,err);pp.append(manual)
  w=np.array([r['weights'][kind] for kind in s.KINDS]);mc.append(sum(wi*pi for wi,pi in zip(w,pp)))
 mb=.9*s.q.independent_affine(a.mixture(P,p.p.ARM),d['y'],a.mixture(raw,p.p.ARM))+.1*short;manual=mb+np.mean(mc,0);err=float(abs(manual-pred).max());assert err<2e-9
 with CSV.open(newline='') as f:r=csv.DictReader(f);assert r.fieldnames==['ID','hardness'];rows=list(r)
 assert [row['ID'] for row in rows]==ids;np.testing.assert_array_equal([float(row['hardness']) for row in rows],pred)
 with CSV.with_suffix('.reproduced.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 assert CSV.read_bytes()==CSV.with_suffix('.reproduced.csv').read_bytes();check();r={'valid':True,'candidate':'018','rows':1000,'finite':True,'ID_order_exact':True,'duplicates':0,'CSV_byte_exact':True,'MPS_reload_exact':True,'CPU_MPS_max_error':cpuerr,'independent_kernel_max_error':maxerr,'independent_affine_weighted_kernel_max_error':err,'CSV_SHA256':e.sha(CSV),'prediction_range':m['prediction_range'],'difference_from017_RMS':m['difference_from017_RMS'],'not_duplicate017':e.sha(CSV)!=e.sha(p.CSV),'best017_preserved':True,'holdout_subgroup_scored':False,'test_fit_or_tuning':False,'new_neural_fits':0,'cost':0,'uploaded':False};e.write(_repo_path(ROOT, 'reports/candidate018-verification.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','fit','training','predict','final']);globals()[parser.parse_args().mode]()
