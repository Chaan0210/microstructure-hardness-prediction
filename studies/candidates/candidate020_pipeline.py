"""Full500 locked156 nonlinear disagreement candidate; best018 stays unchanged."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,csv,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate018_pipeline as old
import disagreement_study_156 as q
import validate_disagreement_v2_156 as validator
import numpy as np,torch,joblib
p=old.p;s=q.s;x=q.x;f=q.f;a=q.a;e=q.e;OUT=_repo_path(ROOT, 'artifacts/final-model-157');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-020.csv')

def prepare():
 old.check();q.check();dev=e.read(q.OUT/'decision.json');assert dev['advance'] and dev['validation_passed'];assert not CSV.exists();OUT.mkdir(exist_ok=False);names=['candidate020_pipeline.py','disagreement_study_156.py','disagreement_features_156.py','validate_disagreement_v2_156.py','artifacts/disagreement-156/decision.json','reports/disagreement-156-summary.json','artifacts/final-model-145/fit.json','artifacts/final-model-145/training-validation.json']
 for split in e.SPLITS:names += [f'artifacts/disagreement-156/validation-{split}.json',f'artifacts/disagreement-156/validation-v2-{split}.json']
 r={'candidate':'020','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'development':dev,'interpretation':'Very small positive pooled margin; two partitions worse, CI includeszero. This is an exploratory artifact under the prespecified gate, not replacement of Publicbest018 or evidence of reliable improvement.','recipe':'Exactly156: unchanged018 base/affines/shortB, no/Ridge585 candidates; RBF586 adds one same-anchor identityCNNmean3-KRR delta. Recompute full500 honestblock candidates and512bootstrap weights with originalseeds.','feature_scope':'For each B166/167, A333/334 fixedcheckpoints produce both B and test features. Never substitute all500 CNN/KRR for testdelta. Validation queries B only in saved trainingfeaturecache, not a new holdout.','reuse':'Existing A-only KRR/CNN3 and all500 base models. No new neuraltraining or affinechange;60residualobjects including12none.','test':'Only inference aftertrainingvalidation; no fitting/selection/clipping changes based on test. No formerholdout subgroup score.','hashes':{g:e.sha(_repo_path(ROOT, g)) for g in names},'protected':q.check()['protected'],'current_best':'018','Publicbest':11.5230652779,'today_submissions_remaining':0,'uploaded':False,'cost':0};e.write(_repo_path(OUT, 'decision.json'),r);e.emit({'locked020':True,'full500_downstream_only':True,'new_neural_fits':0})

def check():
 old.check();q.check();r=e.read(_repo_path(OUT, 'decision.json'))
 for name,h in {**r['hashes'],**r['protected']}.items():assert e.sha(_repo_path(ROOT, name))==h,name
 return r

def inputs(create=False):
 d,P,cells=p.data();new=[]
 for j,(A,B,M,D,short) in enumerate(cells):
  source=_repo_path(ROOT, f'artifacts/final-model-075/inner-{j}');extra=_repo_path(ROOT, f'artifacts/final-model-134/cnn-nested/middle-{j}');z,rec=f.node(d,A,B,B,source,extra,_repo_path(OUT, f'feature-{j}'),create,M);dd=f.augmented(d,z);new.append((dd,A,B,M,D,short,z,rec))
 return d,P,new

def fit():
 check();start=time.perf_counter();d,P,cells=inputs(True);records=[]
 for j,(dd,A,B,M,D,short,z,rec) in enumerate(cells):
  ap,ac=a.affine(a.mixture(D,x.ARM),d['y'][A],a.mixture(M,x.ARM));anchor=.9*ap+.1*short;src=p.OUT/f'block-{j}/residual';r=e.read(src/'report.json');oz=a.readnp(src/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(anchor,oz['anchor_predictions']);path=_repo_path(OUT, f'block-{j}');_,br=q.system(dd,B,anchor,B,a.r.CVSEED+j,s.SEED+j,path);nr=a.readnp(path/'predictions.npz',br['predictions_sha256']);np.testing.assert_array_equal(nr['inner_predictions'][:3],oz['inner_predictions'][:3]);r0=e.read(old.OUT/f'block-{j}/report.json');o0=a.readnp(old.OUT/f'block-{j}/predictions.npz',r0['predictions_sha256']);np.testing.assert_array_equal(nr['full_predictions'][:3],o0['candidate_predictions'][:3,B]);records.append({'block':j,'anchor_calibrator':ac,'feature_report_sha256':e.sha(_repo_path(OUT, f'feature-{j}/report.json')),'residual_report_sha256':e.sha(path/'report.json'),'weights':br['weights']});e.emit({'full500_block':j,'weights':br['weights']})
 r={'rows':500,'train_ids':d['ids'],'blocks':records,'new_residual_objects_including_none':60,'new_neural_fits':0,'linear_predictions_exact018':True,'unchanged018_affines_and_base':True,'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'seconds':time.perf_counter()-start,'holdout_subgroup_scored':False,'test_read':False};e.write(_repo_path(OUT, 'fit.json'),r);check();e.emit({'full500_complete':True,'seconds':r['seconds']})

def training():
 check();d,P,cells=inputs();fit=e.read(_repo_path(OUT, 'fit.json'));assert fit['train_ids']==d['ids'];count=0;maxerr=0.;cpuerr=0.;kerr=0.;covered=[]
 for j,(dd,A,B,M,D,short,z,rec) in enumerate(cells):
  r=fit['blocks'][j];assert e.sha(_repo_path(OUT, f'feature-{j}/report.json'))==r['feature_report_sha256'];path=_repo_path(OUT, f'block-{j}');br=e.read(path/'report.json');assert e.sha(path/'report.json')==r['residual_report_sha256'];saved=a.readnp(path/'predictions.npz',br['predictions_sha256']);anchor=.9*s.q.independent_affine(a.mixture(D,x.ARM),d['y'][A],a.mixture(M,x.ARM))+.1*short;np.testing.assert_allclose(anchor,saved['anchor'],atol=1e-9,rtol=0);_,c,err=validator.verify_system(dd,B,saved['anchor'],B,path);count+=c;maxerr=max(maxerr,err);covered.extend(B.tolist());km=joblib.load(_repo_path(ROOT, rec['models'][0]['path']))['model'];manual=q.vk.manual_kernel(km,d['K'][A],d['y'][A],d['K'][B]);ke=float(abs(manual-z['kernel_B']).max());assert ke<1e-8;kerr=max(kerr,ke)
  for si,item in enumerate(rec['models'][1:]):
   obj=torch.load(_repo_path(ROOT, item['path']),weights_only=True,map_location='cpu');cp=x.predict(obj,d,B,'cpu');ce=float(abs(cp-z['cnn_B'][si]).max());assert ce<1e-3;cpuerr=max(cpuerr,ce)
 assert sorted(covered)==list(range(500));r={'passed':True,'rows':500,'residual_models_verified':count,'independent_residual_max_error':maxerr,'independent_upstream_kernel_max_error':kerr,'CPU_MPS_anchor_CNN_max_error':cpuerr,'all500_honest_block_coverage':True,'same_A_feature_models_B_and_test_locked':True,'unchanged018_training_certificate':e.sha(old.OUT/'training-validation.json'),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'test_read':False,'holdout_subgroup_scored':False};e.write(_repo_path(OUT, 'training-validation.json'),r);check();e.emit(r)

def inference(device='mps'):
 check();v=e.read(_repo_path(OUT, 'training-validation.json'));assert v['passed'] and v['fit_sha256']==e.sha(_repo_path(OUT, 'fit.json'));ids,raw,short,base,_,_,oldpred,X,_=old.inference(device);inp=a.readnp(_repo_path(ROOT, 'artifacts/final-model-053/verification-test-compact-cnn-inputs.npz'));assert inp['ids'].tolist()==ids;td={'images':inp['cnn_images']};query=np.arange(1000);fit=e.read(_repo_path(OUT, 'fit.json'));deltas=[];kernels=[];cnns=[];allpp=[];cor=[]
 for j in range(3):
  rec=e.read(_repo_path(OUT, f'feature-{j}/report.json'));assert e.sha(_repo_path(OUT, f'feature-{j}/report.json'))==fit['blocks'][j]['feature_report_sha256']
  for item in rec['models']:assert e.sha(_repo_path(ROOT, item['path']))==item['sha256']
  kp=joblib.load(_repo_path(ROOT, rec['models'][0]['path']))['model'].predict(inp['compact']);cp=[]
  for item in rec['models'][1:]:cp.append(x.predict(torch.load(_repo_path(ROOT, item['path']),weights_only=True,map_location='cpu'),td,query,device))
  cp=np.stack(cp);delta=cp.mean(0)-kp;XX=np.column_stack([X,delta]);path=_repo_path(OUT, f'block-{j}');br=e.read(path/'report.json');assert e.sha(path/'report.json')==fit['blocks'][j]['residual_report_sha256'];w=np.array([br['weights'][kind] for kind in s.KINDS]);pp=[]
  for kind in s.KINDS:
   file=path/f'full-{kind}.joblib';assert e.sha(file)==br['models_sha256'][file.name];pp.append(q.predict(joblib.load(_repo_io(file))['model'],XX))
  pp=np.stack(pp);deltas.append(delta);kernels.append(kp);cnns.append(cp);allpp.append(pp);cor.append(w@pp)
 cor=np.stack(cor);pred=base+cor.mean(0);assert pred.shape==(1000,) and np.isfinite(pred).all();return ids,raw,short,base,np.stack(allpp),cor,pred,X,np.stack(deltas),np.stack(kernels),np.stack(cnns),oldpred

def predict():
 ids,raw,short,base,pp,cor,pred,X,delta,kp,cp,oldpred=inference();assert len(ids)==len(set(ids))==1000
 with CSV.open('x',newline='') as f0:w=csv.writer(f0);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f0:np.savez_compressed(_repo_io(f0),ids=np.array(ids),components=raw,short=short,base=base,candidate_predictions=pp,corrections=cor,primary=pred,delta=delta,anchor_kernel=kp,anchor_cnn=cp,reference018=oldpred)
 r={'candidate':'020','rows':1000,'submission_sha256':e.sha(CSV),'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'training_validation_sha256':e.sha(_repo_path(OUT, 'training-validation.json')),'sample_submission_sha256':e.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'finite':True,'ID_order_exact':True,'duplicates':0,'prediction_range':[float(pred.min()),float(pred.max())],'difference_from018_RMS':float(np.sqrt(np.mean((pred-oldpred)**2))),'feature_source':'Three separate A333/334 model bundles, not all500 outer base models.','output_clipping':False,'feature_clipping':'Train-fitted MinMax clip preserved','test_fit_or_tuning':False,'holdout_subgroup_scored':False,'uploaded':False,'cost':0};assert e.sha(CSV)!=e.sha(old.CSV);e.write(CSV.with_suffix('.manifest.json'),r);e.emit(r)

def final():
 ids,raw,short,base,pp,cor,pred,X,delta,kp,cp,oldpred=inference();m=e.read(CSV.with_suffix('.manifest.json'));assert e.sha(CSV)==m['submission_sha256'];z=a.readnp(_repo_path(OUT, 'predictions.npz'),m['predictions_sha256']);assert z['ids'].tolist()==ids
 for key,arr in [('components',raw),('short',short),('base',base),('candidate_predictions',pp),('corrections',cor),('primary',pred),('delta',delta),('anchor_kernel',kp),('anchor_cnn',cp),('reference018',oldpred)]:np.testing.assert_array_equal(z[key],arr)
 ids2,_,_,_,_,_,cpu,_,_,_,_,_=inference('cpu');assert ids2==ids;cpuerr=float(abs(cpu-pred).max());assert cpuerr<1e-3;d,P,cells=inputs();manualcor=[];maxerr=0.;upstreamerr=0.;inp=a.readnp(_repo_path(ROOT, 'artifacts/final-model-053/verification-test-compact-cnn-inputs.npz'))
 for j,(dd,A,B,M,D,sh,fz,rec) in enumerate(cells):
  km=joblib.load(_repo_path(ROOT, rec['models'][0]['path']))['model'];manualK=q.vk.manual_kernel(km,d['K'][A],d['y'][A],inp['compact']);ke=float(abs(manualK-kp[j]).max());assert ke<1e-8;upstreamerr=max(upstreamerr,ke);np.testing.assert_array_equal(cp[j].mean(0)-kp[j],delta[j]);XX=np.column_stack([X,delta[j]]);path=_repo_path(OUT, f'block-{j}');br=e.read(path/'report.json');bz=a.readnp(path/'predictions.npz',br['predictions_sha256']);mp=[]
  for ci,kind in enumerate(s.KINDS):
   model=joblib.load(_repo_io(path/f'full-{kind}.joblib'))['model'];v=q.manual(model,dd['X'][B],bz['residual'],XX,dd['blocks']);err=float(abs(v-pp[j,ci]).max());assert err<2e-9;maxerr=max(maxerr,err);mp.append(v)
  weights=np.array([br['weights'][kind] for kind in s.KINDS]);manualcor.append(sum(wi*pi for wi,pi in zip(weights,mp)))
 manual=.9*s.q.independent_affine(a.mixture(P,x.ARM),d['y'],a.mixture(raw,x.ARM))+.1*short+np.mean(manualcor,0);err=float(abs(manual-pred).max());assert err<2e-9
 with CSV.open(newline='') as f0:reader=csv.DictReader(f0);assert reader.fieldnames==['ID','hardness'];rows=list(reader)
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal([float(r['hardness']) for r in rows],pred)
 with CSV.with_suffix('.reproduced.csv').open('x',newline='') as f0:w=csv.writer(f0);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 assert CSV.read_bytes()==CSV.with_suffix('.reproduced.csv').read_bytes();check();r={'valid':True,'candidate':'020','rows':1000,'finite':True,'ID_order_exact':True,'duplicates':0,'CSV_byte_exact':True,'MPS_reload_exact':True,'CPU_MPS_max_error':cpuerr,'independent_residual_max_error':maxerr,'independent_upstream_KRR_max_error':upstreamerr,'independent_affine_weighted_residual_max_error':err,'CSV_SHA256':e.sha(CSV),'prediction_range':m['prediction_range'],'difference_from018_RMS':m['difference_from018_RMS'],'same_A_scope_B_test_verified':True,'not_duplicate018':e.sha(CSV)!=e.sha(old.CSV),'best018_preserved':True,'holdout_subgroup_scored':False,'test_fit_or_tuning':False,'new_neural_fits':0,'cost':0,'uploaded':False};e.write(_repo_path(ROOT, 'reports/candidate020-verification.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','fit','training','predict','final']);globals()[parser.parse_args().mode]()
