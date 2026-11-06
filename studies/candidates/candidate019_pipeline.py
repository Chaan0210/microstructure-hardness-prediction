"""Locked151 full500 D4 CNN inference at every nested level of018."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,csv,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_view_mean_151 as q
import candidate018_pipeline as old
import numpy as np,torch,joblib
p=old.p;s=q.s;x=q.x;a=q.a;e=q.e;core=q.core;n=s.n
OUT=_repo_path(ROOT, 'artifacts/final-model-152');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-019.csv')

def prepare():
 old.check();q.check();dev=e.read(q.OUT/'decision.json');assert dev['advance'] and dev['validation_passed'];assert e.read(q.OUT/'extra-validation.json')['passed'];assert not CSV.exists();OUT.mkdir(exist_ok=False)
 names=['candidate019_pipeline.py','cnn_view_mean_151.py','validate_cnn_view_extra_151.py','directional_energy_148.py','artifacts/cnn-view-mean-151/decision.json','artifacts/cnn-view-mean-151/extra-validation.json','reports/cnn-view-mean-151-summary.json','reports/candidate018-public-submission.json','artifacts/final-model-145/fit.json','artifacts/final-model-145/training-validation.json','reports/nested-selector-146-audit.json']
 names += [f'artifacts/cnn-view-mean-151/validation-{v}.json' for v in e.SPLITS]
 protected={**q.check()['protected'],'deliverables/reproduction-011-014-015-v136.zip':e.sha(_repo_path(ROOT, 'deliverables/reproduction-011-014-015-v136.zip'))}
 r={'candidate':'019','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'development':dev,'recipe':'Exactly151: uniform3seed x8D4 CNN means at outer/full500, middle and deep levels; .3KRR/.3CNN/.2S/.2B; refit outer and three anchor affines; .9calibrated+.1shortB; physics57+adjacency528 bootstrap-weighted five-candidate residuals,mean3honestblocks. No148/149features or tuneable blend.','full500':'Reuse all existing base checkpoints. Compute missing12nestedD4nodes (36existingCNNcheckpoints). Refit4affines and60residualobjects including12none;48nonnull fits. Identical3innerfolds,512groupbootstrapdraws,seed20271204+j.','holdout':'All500training afterdevelopment lock only, no formerholdout subgroup scoring.','test':'Inference only afternewtraining validation. No test fitting, selection, output clipping or blend changes.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':protected,'current_Publicbest':'018','current_Publicbest_score':11.5230652779,'new_neural_fits':0,'cost':0,'uploaded':False};e.write(_repo_path(OUT, 'decision.json'),r);e.emit({'candidate019_locked':True,'new_neural_fits':0,'nested_CNN_predictions':36,'new_nonnull_residual_fits':48})

def check():
 old.check();q.check();r=e.read(_repo_path(OUT, 'decision.json'))
 for f,h in {**r['hashes'],**r['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 return r

def inputs(create=False):
 d,P,cells=p.data();P=P.copy();result=[];root=_repo_path(OUT, 'cnn-nested')
 if create:root.mkdir(exist_ok=False)
 for j,(A,B,M,D,short) in enumerate(cells):
  M=M.copy();D=D.copy();oldroot=p.p.OUT/f'cnn-nested/middle-{j}';r=e.read(oldroot/'report.json');z=a.readnp(oldroot/'predictions.npz',r['predictions_sha256']);models=[_repo_path(ROOT, f'artifacts/final-model-075/inner-{j}/cnn.pt'),*[oldroot/f'seed-{seed}.pt' for seed in x.SEEDS[1:]]];M[:,2],_=q.node(d,A,B,models,z['seeds'],root/f'middle-{j}',create);P[B,2]=M[:,2];lookup={int(i):ii for ii,i in enumerate(A)}
  for ell in range(3):
   source=_repo_path(ROOT, f'artifacts/final-model-114/block-{j}/deep-{ell}');C,V,_=a.bundle(source);assert set(C)|set(V)==set(A) and not set(C)&set(V) and not(set(C)|set(V))&set(B);oldroot=p.p.OUT/f'cnn-nested/deep-{j}-{ell}';r=e.read(oldroot/'report.json');z=a.readnp(oldroot/'predictions.npz',r['predictions_sha256']);models=[source/'cnn.pt',*[oldroot/f'seed-{seed}.pt' for seed in x.SEEDS[1:]]];pp,_=q.node(d,C,V,models,z['seeds'],root/f'deep-{j}-{ell}',create);D[[lookup[int(i)] for i in V],2]=pp
  assert not set(d['g'][A])&set(d['g'][B]);result.append((A,B,M,D,short));e.emit({'full500_D4_block':j,'create':create})
 assert np.isfinite(P).all();return d,P,result

def fit():
 check();start=time.perf_counter();d,P,cells=inputs(True);mix=a.mixture(P,x.ARM);_,cal=a.affine(mix,d['y'],mix);e.write(_repo_path(OUT, 'calibrator.json'),{**cal,'n_fit':500,'train_ids':d['ids']})
 with (_repo_path(OUT, 'calibration_training.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),components=P,inner_oof=mix)
 records=[]
 for j,(A,B,M,D,short) in enumerate(cells):
  path=_repo_path(OUT, f'block-{j}');path.mkdir(exist_ok=False);ap,ac=a.affine(a.mixture(D,x.ARM),d['y'][A],a.mixture(M,x.ARM));anchor=.9*ap+.1*short;_,rec=core.system(d,B,anchor,np.arange(500),a.r.CVSEED+j,s.SEED+j,path/'residual')
  with (path/'anchor.npz').open('xb') as f:np.savez_compressed(_repo_io(f),anchor_indices=A,block_indices=B,deep_components=D,middle_components=M,short=short,anchor=anchor)
  r={'block':j,'anchor_calibrator':ac,'anchor_sha256':e.sha(path/'anchor.npz'),'residual_report_sha256':e.sha(path/'residual/report.json'),'weights':rec['weights']};e.write(path/'report.json',r);records.append(r);e.emit({'full500_block':j,'weights':rec['weights']})
 r={'rows':500,'train_ids':d['ids'],'blocks':records,'new_affine_fits':4,'new_residual_objects_including_none':60,'new_nonnull_residual_fits':48,'new_neural_fits':0,'calibrator_sha256':e.sha(_repo_path(OUT, 'calibrator.json')),'calibration_training_sha256':e.sha(_repo_path(OUT, 'calibration_training.npz')),'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'seconds':time.perf_counter()-start,'holdout_subgroup_scored':False,'test_read':False};e.write(_repo_path(OUT, 'fit.json'),r);check();e.emit({'full500_fit_complete':True,'seconds':r['seconds']})

def training():
 check();d,P,cells=inputs();fit=e.read(_repo_path(OUT, 'fit.json'));assert fit['train_ids']==d['ids'];assert e.sha(_repo_path(OUT, 'calibrator.json'))==fit['calibrator_sha256'];z=a.readnp(_repo_path(OUT, 'calibration_training.npz'),fit['calibration_training_sha256']);np.testing.assert_array_equal(P,z['components']);assert z['ids'].tolist()==d['ids'];cal=e.read(_repo_path(OUT, 'calibrator.json'));mix=a.mixture(P,x.ARM);manual=s.q.independent_affine(mix,d['y'],mix);np.testing.assert_allclose(manual,cal['target_center']+cal['slope']*(mix-cal['prediction_center']),atol=1e-9,rtol=0);covered=[];count=0;maxerr=0.;cpuerr=0.
 for j,(A,B,M,D,short) in enumerate(cells):
  path=_repo_path(OUT, f'block-{j}');r=e.read(path/'report.json');assert r==fit['blocks'][j];z=a.readnp(path/'anchor.npz',r['anchor_sha256']);np.testing.assert_array_equal(z['anchor_indices'],A);np.testing.assert_array_equal(z['block_indices'],B);np.testing.assert_array_equal(z['middle_components'],M);np.testing.assert_array_equal(z['deep_components'],D);ap,ac=a.affine(a.mixture(D,x.ARM),d['y'][A],a.mixture(M,x.ARM));assert ac==r['anchor_calibrator'];anchor=.9*ap+.1*short;np.testing.assert_array_equal(anchor,z['anchor']);np.testing.assert_allclose(anchor,.9*s.q.independent_affine(a.mixture(D,x.ARM),d['y'][A],a.mixture(M,x.ARM))+.1*short,atol=1e-9,rtol=0);assert e.sha(path/'residual/report.json')==r['residual_report_sha256'];_,c,err=core.verify_system(d,B,anchor,np.arange(500),path/'residual');count+=c;maxerr=max(maxerr,err);covered.extend(B.tolist())
 assert sorted(covered)==list(range(500))
 for path in [_repo_path(OUT, 'cnn-nested/middle-0'),_repo_path(OUT, 'cnn-nested/deep-0-0')]:
  r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256'])
  for si,m in enumerate(r['models']):
   vv=q.views.predict_views(d['images'][z['validation']],_repo_path(ROOT, m['path']),'cpu');err=float(abs(vv-z['views'][si]).max());assert err<1e-3;cpuerr=max(cpuerr,err)
 for seed in x.SEEDS:
  obj=torch.load(_repo_io(p.p.cnnpath(seed)),map_location='cpu',weights_only=True);assert obj['train_ids']==d['ids'];assert obj['normalization']=={'x_mean':float(d['images'].mean()),'x_std':float(d['images'].std()),'y_mean':float(d['y'].mean()),'y_std':float(d['y'].std())}
 r={'passed':True,'rows':500,'residual_models_verified':count,'independent_max_error':maxerr,'all500_honest_block_coverage':True,'nested_D4_nodes_verified':12,'CPU_MPS_nested_view_max_error':cpuerr,'full500_CNN_train_ids_and_normalization_exact':True,'unchanged_base_training_certificate':e.sha(p.p.OUT/'training-validation.json'),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'holdout_subgroup_scored':False,'test_read':False};e.write(_repo_path(OUT, 'training-validation.json'),r);check();e.emit(r)

def inference(device='mps'):
 check();v=e.read(_repo_path(OUT, 'training-validation.json'));assert v['passed'] and v['fit_sha256']==e.sha(_repo_path(OUT, 'fit.json'));ids,raw,short,_,_,_,X=p.inference(device);z=a.readnp(_repo_path(ROOT, 'artifacts/final-model-053/verification-test-compact-cnn-inputs.npz'));assert z['ids'].tolist()==ids;vv=np.stack([q.views.predict_views(z['cnn_images'],p.p.cnnpath(seed),device) for seed in x.SEEDS]);np.testing.assert_allclose(vv[:,0].mean(0),raw[:,2],atol=1e-4,rtol=0);raw=raw.copy();raw[:,2]=vv.mean((0,1));fit=e.read(_repo_path(OUT, 'fit.json'));assert e.sha(_repo_path(OUT, 'calibrator.json'))==fit['calibrator_sha256'];cal=e.read(_repo_path(OUT, 'calibrator.json'));base=.9*(cal['target_center']+cal['slope']*(a.mixture(raw,x.ARM)-cal['prediction_center']))+.1*short;allpp=[];cor=[]
 for j in range(3):
  root=_repo_path(OUT, f'block-{j}/residual');r=e.read(root/'report.json');assert e.sha(root/'report.json')==fit['blocks'][j]['residual_report_sha256'];w=np.array([r['weights'][kind] for kind in s.KINDS]);pp=[]
  for kind in s.KINDS:
   file=root/f'full-{kind}.joblib';assert e.sha(file)==r['models_sha256'][file.name];pp.append(n.predict(joblib.load(_repo_io(file))['model'],X))
  pp=np.stack(pp);allpp.append(pp);cor.append(w@pp)
 cor=np.stack(cor);pred=base+cor.mean(0);assert pred.shape==(1000,) and np.isfinite(pred).all();return ids,raw,short,base,np.stack(allpp),cor,pred,X,vv

def predict():
 ids,raw,short,base,allpp,cor,pred,X,vv=inference();assert len(ids)==len(set(ids))==1000
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),components=raw,short=short,base=base,candidate_predictions=allpp,corrections=cor,primary=pred,cnn_views=vv)
 oldm=e.read(old.CSV.with_suffix('.manifest.json'));oz=a.readnp(old.OUT/'predictions.npz',oldm['predictions_sha256']);assert oz['ids'].tolist()==ids;r={'candidate':'019','rows':1000,'submission_sha256':e.sha(CSV),'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'training_validation_sha256':e.sha(_repo_path(OUT, 'training-validation.json')),'sample_submission_sha256':e.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'ID_order_exact':True,'finite':True,'duplicates':0,'prediction_range':[float(pred.min()),float(pred.max())],'difference_from018_RMS':float(np.sqrt(np.mean((pred-oz['primary'])**2))),'output_clipping':False,'feature_clipping':'Existing train-fitted MinMax clip only','holdout_subgroup_scored':False,'test_fit_or_tuning':False,'uploaded':False,'cost':0};assert r['submission_sha256']!=e.sha(old.CSV);e.write(CSV.with_suffix('.manifest.json'),r);e.emit(r)

def final():
 ids,raw,short,base,allpp,cor,pred,X,vv=inference();m=e.read(CSV.with_suffix('.manifest.json'));assert e.sha(CSV)==m['submission_sha256'];z=a.readnp(_repo_path(OUT, 'predictions.npz'),m['predictions_sha256']);assert z['ids'].tolist()==ids
 for key,arr in [('components',raw),('short',short),('base',base),('candidate_predictions',allpp),('corrections',cor),('primary',pred),('cnn_views',vv)]:np.testing.assert_array_equal(arr,z[key])
 ids2,_,_,_,_,_,cpu,_,_=inference('cpu');assert ids2==ids;cpuerr=float(abs(cpu-pred).max());assert cpuerr<1e-3;d,P,cells=inputs();mc=[];maxerr=0.
 for j,(A,B,M,D,sh) in enumerate(cells):
  root=_repo_path(OUT, f'block-{j}/residual');r=e.read(root/'report.json');rz=a.readnp(root/'predictions.npz',r['predictions_sha256']);pp=[]
  for ci,kind in enumerate(s.KINDS):
   model=joblib.load(_repo_io(root/f'full-{kind}.joblib'))['model'];manual=s.q.vn.manual(model,d['X'][B],rz['residual'],X,d['blocks']);err=float(abs(manual-allpp[j,ci]).max());assert err<2e-9;maxerr=max(maxerr,err);pp.append(manual)
  w=np.array([r['weights'][kind] for kind in s.KINDS]);mc.append(sum(wi*pi for wi,pi in zip(w,pp)))
 manual=.9*s.q.independent_affine(a.mixture(P,x.ARM),d['y'],a.mixture(raw,x.ARM))+.1*short+np.mean(mc,0);err=float(abs(manual-pred).max());assert err<2e-9
 with CSV.open(newline='') as f:reader=csv.DictReader(f);assert reader.fieldnames==['ID','hardness'];rows=list(reader)
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal([float(r['hardness']) for r in rows],pred)
 with CSV.with_suffix('.reproduced.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 assert CSV.read_bytes()==CSV.with_suffix('.reproduced.csv').read_bytes();check();r={'valid':True,'candidate':'019','rows':1000,'finite':True,'ID_order_exact':True,'duplicates':0,'CSV_byte_exact':True,'MPS_reload_exact':True,'CPU_MPS_max_error':cpuerr,'independent_kernel_max_error':maxerr,'independent_affine_weighted_kernel_max_error':err,'CSV_SHA256':e.sha(CSV),'prediction_range':m['prediction_range'],'difference_from018_RMS':m['difference_from018_RMS'],'not_duplicate018':e.sha(CSV)!=e.sha(old.CSV),'best018_preserved':True,'holdout_subgroup_scored':False,'test_fit_or_tuning':False,'new_neural_fits':0,'cost':0,'uploaded':False};e.write(_repo_path(ROOT, 'reports/candidate019-verification.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','fit','training','predict','final']);globals()[parser.parse_args().mode]()
