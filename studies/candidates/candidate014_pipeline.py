"""Locked no-baseline795 procedure: full500 nested downstream fit, then test inference."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,csv,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import component_single_ablation_122 as study
import candidate012_residual_pipeline as c
import candidate008_pipeline as p8
import validate_nonlinear_residual_115 as vn
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
a=study.a;e=study.e;n=study.n;r=study.r;OUT=_repo_path(ROOT, 'artifacts/final-model-123');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-014.csv');ARM='no_baseline795'

def inputs():
 d=c.data();d['X']=c.getX();d['blocks']=r.load('original')['blocks'];d['fid']=np.zeros(500,int);P=np.full((500,5),np.nan);cells=[];root75=e.read(c.p9.OUT/'report.json');root114=e.read(c.OUT/'fit.json');seen=[]
 for j,(A,B) in enumerate(GroupKFold(3,shuffle=True,random_state=e.cal.INNER_SEED).split(d['S'],d['y'],d['g'])):
  middle=c.p9.OUT/f'inner-{j}';assert e.sha(middle/'report.json')==root75['inner_report_sha256'][f'inner-{j}/report.json'];AA,BB,M=a.bundle(middle);np.testing.assert_array_equal(AA,A);np.testing.assert_array_equal(BB,B);P[B]=M;seen.extend(B.tolist());path=c.OUT/f'block-{j}';br=e.read(path/'report.json');assert br==root114['blocks'][j];z=a.readnp(path/'anchor.npz',br['anchor_sha256']);np.testing.assert_array_equal(z['train_indices'],A);np.testing.assert_array_equal(z['residual_indices'],B);D=np.full((len(A),5),np.nan);lookup={int(i):k for k,i in enumerate(A)};deepseen=[]
  for ell in range(3):
   deep=path/f'deep-{ell}';assert e.sha(deep/'report.json')==br['deep_report_sha256'][f'deep-{ell}/report.json'];it,iv,Q=a.bundle(deep);assert set(it)|set(iv)==set(A) and not set(it)&set(iv) and not(set(it)|set(iv))&set(B);D[[lookup[int(i)] for i in iv]]=Q;deepseen.extend(iv.tolist())
  assert sorted(deepseen)==sorted(A.tolist());assert not set(d['g'][A])&set(d['g'][B]);sr=e.read(path/'shortB/report.json');assert e.sha(path/'shortB/report.json')==br['shortB_report_sha256'];assert e.sha(path/'shortB/model.pt')==sr['model_sha256'];s=a.readnp(path/'shortB/predictions.npz',sr['predictions_sha256']);np.testing.assert_array_equal(s['validation_indices'],B);np.testing.assert_array_equal(s['predictions'],z['shortB']);cells.append((A,B,M,D,z['shortB']))
 assert sorted(seen)==list(range(500)) and np.isfinite(P).all();return d,P,cells

def prepare():
 a.check();decision=e.read(study.OUT/'decision.json');assert decision['eligible'] and decision['selected_configuration']==ARM;assert not CSV.exists();OUT.mkdir(exist_ok=False);assert e.read(c.OUT/'training-validation.json')['passed'];assert e.read(c.OUT/'training-validation.json')['fit_sha256']==e.sha(c.OUT/'fit.json');d,P,cells=inputs();sourcefiles={}
 for folder in [c.p9.OUT,*[c.OUT/f'block-{j}' for j in range(3)]]:
  for q in folder.rglob('*'):
   if q.is_file():sourcefiles[str(q.relative_to(ROOT))]=e.sha(q)
 for name in ['artifacts/final-model-114/train57.npz','artifacts/final-model-114/train57.json','artifacts/final-model-114/test57.npz','artifacts/final-model-114/test57.json','artifacts/final-model-114/fit.json','artifacts/final-model-114/training-validation.json','artifacts/final-model-017/report.json','artifacts/final-model-017/kernel.joblib','artifacts/final-model-017/cnn.pt','artifacts/final-model-065/report.json','artifacts/final-model-086/report.json','artifacts/final-model-086/head/model.pt','artifacts/final-model-053/verification-test-compact-cnn-inputs.npz','reports/candidate008-verification.json','reports/candidate011-verification.json','reports/candidate013-verification.json','artifacts/submission-candidate-008.components.csv','artifacts/submission-candidate-008.manifest.json','artifacts/submission-candidate-011.components.csv','artifacts/submission-candidate-011.manifest.json','artifacts/submission-candidate-013.csv','artifacts/submission-candidate-013.manifest.json']:sourcefiles[name]=e.sha(_repo_path(ROOT, name))
 m8=e.read(_repo_path(ROOT, 'artifacts/submission-candidate-008.manifest.json'));fit8=e.read(_repo_path(ROOT, 'artifacts/final-model-065/report.json'))
 for rec in fit8['models']:path=p8.OUT/rec['checkpoint'];assert e.sha(path)==rec['checkpoint_sha256'];sourcefiles[str(path.relative_to(ROOT))]=e.sha(path)
 for arm in ['S','B']:
  for metadata,prefix in [(fit8,'train'),(m8,'test')]:path=_repo_path(ROOT, metadata[f'{prefix}_{arm}_cache']);assert e.sha(path)==metadata[f'{prefix}_{arm}_sha256'];sourcefiles[str(path.relative_to(ROOT))]=e.sha(path)
 p={'candidate':'014','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'development_decision':decision,'selected_arm':ARM,'formula':'mixture=.3compact26KRR+.3TinyCNN+.2Smean3+.2Bmean3; base=.9*newaffine(mixture)+.1Bshort; prediction=base+mean3newhonestresidualcorrections','procedure':'Same122algorithm. Refitoneaffineon500OOFpairs;3anchor-affineson333/334deepOOFpairs;refitfivewayresidualselectorandchosenmodelineach166/167honestblock. Reuseindependentremainingbaselearnersalreadytrainedall500andverified075/114nestedbasepredictions. No remainingbasearchitecture/hyperparameterschanged.','neural_retraining':'Notrequiredorperformed:remainingbranchesindependentofremovedbaseline. Thisisfull500downstreamretrainingofchangedpipeline,notclaimofnewneuralfits.','train_scope':'All500maytrainafterlock;formerholdoutnotseparatelyscored. OOF500usedforfittingcalibrator,notreportedvalidation.','test_scope':'Hashcachedtestassetsforintegritynow;decodetestfeatures/inferonlyafterfitandtrainingvalidation. No testfit,selection,thresholdorclipping.','sources':{name:e.sha(_repo_path(ROOT, name)) for name in ['candidate014_pipeline.py','validate_candidate014.py','component_single_ablation_122.py','component_ablation_121.py','nonlinear_residual_115.py']},'inputs':sourcefiles,'development_decision_sha256':e.sha(study.OUT/'decision.json'),'protected':a.check()['protected'],'cost':0,'uploaded':False};e.write(_repo_path(OUT, 'decision.json'),p);e.emit({'candidate014_locked':True,'n_train':500,'mixture':[0,.3,.3,.2,.2]})

def check():
 a.check();p=e.read(_repo_path(OUT, 'decision.json'));assert e.sha(study.OUT/'decision.json')==p['development_decision_sha256']
 for name,h in {**p['sources'],**p['inputs'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, name))==h,name
 return p

def fit():
 check();d,P,cells=inputs();start=time.perf_counter();ip=a.mixture(P,ARM);_,cal=a.affine(ip,d['y'],ip);e.write(_repo_path(OUT, 'calibrator.json'),{**cal,'n_fit':500,'train_ids':d['ids']})
 with (_repo_path(OUT, 'calibration_training.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),components=P,inner_oof=ip)
 records=[]
 for j,(A,B,M,D,short) in enumerate(cells):
  path=_repo_path(OUT, f'block-{j}');path.mkdir(exist_ok=False);pred,coef=a.affine(a.mixture(D,ARM),d['y'][A],a.mixture(M,ARM));anchor=.9*pred+.1*short;cor,rec=a.fit_block(d,B,anchor,0,j,path/'residual')
  with (path/'anchor.npz').open('xb') as f:np.savez_compressed(_repo_io(f),anchor_indices=A,block_indices=B,deep_components=D,middle_components=M,short=short,anchor=anchor)
  record={'block':j,'calibrator':coef,'anchor_rows':len(A),'residual_rows':len(B),'chosen':rec['chosen'],'anchor_sha256':e.sha(path/'anchor.npz'),'residual_report_sha256':e.sha(path/'residual/report.json')};e.write(path/'report.json',record);records.append(record);e.emit({'full500_block':j,'chosen':rec['chosen']})
 report={'n_train':500,'train_ids':d['ids'],'new_affine_fits':4,'new_residual_blocks':3,'new_residual_models_including_none':48,'new_neural_fits':0,'remaining_base_models_already_trained_on500':True,'train_csv_sha256':e.sha(_repo_path(ROOT, 'data/train.csv')),'calibrator_sha256':e.sha(_repo_path(OUT, 'calibrator.json')),'calibration_training_sha256':e.sha(_repo_path(OUT, 'calibration_training.npz')),'blocks':records,'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'seconds':time.perf_counter()-start,'holdout_subgroup_scored':False,'test_decoded':False};e.write(_repo_path(OUT, 'fit.json'),report);check();e.emit({'full500_downstream_fit_complete':True,'seconds':report['seconds']})

def inference(device='mps'):
 check();v=e.read(_repo_path(OUT, 'training-validation.json'));assert v['passed'] and v['fit_sha256']==e.sha(_repo_path(OUT, 'fit.json'));ids,_=c.cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);m8=e.read(_repo_path(ROOT, 'artifacts/submission-candidate-008.manifest.json'));model8=e.read(p8.OUT/'report.json');features={};heads={}
 for arm in ['S','B']:
  path=_repo_path(ROOT, m8[f'test_{arm}_cache']);assert e.sha(path)==m8[f'test_{arm}_sha256'];z=a.readnp(path);assert z['ids'].tolist()==ids;features[arm]=z['X'].astype(float);pp=[]
  for rec in model8['models']:
   if rec['arm']==arm:pp.append(p8.predict_checkpoint(p8.OUT/rec['checkpoint'],features[arm]))
  heads[arm]=np.mean(pp,axis=0)
 z=a.readnp(_repo_path(ROOT, 'artifacts/final-model-053/verification-test-compact-cnn-inputs.npz'));assert z['ids'].tolist()==ids;kernel=joblib.load(_repo_path(ROOT, 'artifacts/final-model-017/kernel.joblib'));kp=kernel['model'].predict(z['compact']);cnn=c.cp.predict_cnn(z['cnn_images'],_repo_path(ROOT, 'artifacts/final-model-017/cnn.pt'),device);raw=np.column_stack([np.zeros(1000),kp,cnn,heads['S'],heads['B']]);mixture=a.mixture(raw,ARM);cal=e.read(_repo_path(OUT, 'calibrator.json'));calpred=cal['target_center']+cal['slope']*(mixture-cal['prediction_center']);saved=torch.load(_repo_path(ROOT, 'artifacts/final-model-086/head/model.pt'),weights_only=True,map_location='cpu');small=e.Small(saved['scaler_mean'].numpy(),saved['scaler_scale'].numpy(),20270215);small.load_state_dict({k:saved['state'][k] for k in small.state_dict()});small=small.to(device);short=e.small_predict(small,features['B'],np.arange(1000),device=device)*saved['target_std']+saved['target_mean'];del small;torch.mps.empty_cache();base=.9*calpred+.1*short
 z=a.readnp(c.OUT/'test57.npz',e.read(c.OUT/'test57.json')['sha256']);assert z['ids'].tolist()==ids;X=z['X'];cor=[]
 for j in range(3):
  path=_repo_path(OUT, f'block-{j}/residual');rec=e.read(path/'report.json');assert e.sha(path/'selected.joblib')==rec['models_sha256']['selected.joblib'];cor.append(n.predict(joblib.load(_repo_io(path/'selected.joblib'))['model'],X))
 cor=np.stack(cor);pred=base+cor.mean(0);assert pred.shape==(1000,) and np.isfinite(pred).all();return ids,raw,short,base,cor,pred,X

def predict():
 ids,raw,short,base,cor,pred,X=inference();assert len(set(ids))==1000
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','kernel','cnn','Smean','Bmean','shortB','new_base','residual0','residual1','residual2','candidate014']);w.writerows(zip(ids,*raw[:,1:].T,short,base,*cor,pred))
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),components=raw,short=short,base=base,corrections=cor,primary=pred)
 fit=e.read(_repo_path(OUT, 'fit.json'));lo,hi=e.read(p8.OUT/'report.json')['training_label_range'];manifest={'candidate':'014','rows':1000,'columns':['ID','hardness'],'submission_sha256':e.sha(CSV),'components_sha256':e.sha(CSV.with_suffix('.components.csv')),'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'training_validation_sha256':e.sha(_repo_path(OUT, 'training-validation.json')),'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'sample_submission_sha256':e.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'selected':[v['chosen'] for v in fit['blocks']],'prediction_range':[float(pred.min()),float(pred.max())],'training_label_range':[lo,hi],'outside_training_range':int(((pred<lo)|(pred>hi)).sum()),'finite':True,'duplicates':0,'ID_order_exact':True,'inference':'Freshreloadremainingfull500KRR,CNN,S/Bheads,shortB,andnewcalibrator/residuals;no795baselineused.','clipping':False,'test_fit_or_tuning':False,'holdout_subgroup_scored':False,'013_preserved':True,'uploaded':False,'cost':0};e.write(CSV.with_suffix('.manifest.json'),manifest);check();e.emit({'CSV_ready':True,'SHA256':e.sha(CSV),'range':manifest['prediction_range']})
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit','predict']);x=p.parse_args();globals()[x.mode]()
