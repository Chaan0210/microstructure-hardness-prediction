"""Locked no-baseline795 procedure: full500 nested downstream fit, then test inference."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,csv,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate014_pipeline as previous
import cnn_seed_mean_133 as x
import component_single_ablation_122 as study
import candidate012_residual_pipeline as c
import candidate008_pipeline as p8
import validate_nonlinear_residual_115 as vn
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
a=study.a;e=study.e;n=study.n;r=study.r;OUT=_repo_path(ROOT, 'artifacts/final-model-134');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-015.csv');ARM='no_baseline795'


CNN_ROOT=_repo_path(ROOT, 'artifacts/final-model-024')
def cnnpath(seed):return _repo_path(CNN_ROOT, f'cnn-seed{seed}.pt')
def prepare():
 previous.check();x.check();assert not CSV.exists();OUT.mkdir(exist_ok=False)
 dev=e.read(x.OUT/'confirmation-decision.json');assert dev['all_independent_validation_passed'] and dev['lower_pooled_MSE']
 names=['candidate015_pipeline.py','validate_candidate015.py','candidate014_pipeline.py','validate_candidate014.py','cnn_seed_mean_133.py','validate_cnn_seed_mean_133.py','candidate004_pipeline.py','frozen_validation.py','reports/cnn-seed-mean-133-summary.json','reports/cnn-seed-mean-133-preservation.json','artifacts/cnn-seed-mean-133/confirmation-decision.json','artifacts/cnn-seed-mean-133/pooled-decision-policy.json','artifacts/final-model-024/report.json','artifacts/submission-candidate-014.csv','artifacts/submission-candidate-014.manifest.json']
 names += [str(cnnpath(seed).relative_to(ROOT)) for seed in x.SEEDS]
 p={'candidate':'015','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'Parent explicitly authorized fixed133 recipe full500 fitting and test1000 inference, no upload.','fixed_seeds':list(x.SEEDS),'formula':'raw=.3compact26KRR+.3mean3identityCNN+.2Smean3+.2Bmean3; base=.9*newaffine(raw)+.1shortB; prediction=base+mean3newhonestphysics57residuals','weights_changed':False,'seed_selection':False,'inference_augmentation':'identity only; historical024 D4 inference is not reused','development':dev,'pooled_RMSE_gain':0.0355203045085446,'conditional95_gain_CI':[-0.02680458603553579,0.09978930199665262],'limits':['Same400 adaptively reused; overlapping partitions are not independent.','Fixed-fit CI excludes model refitting and research-selection uncertainty; includes zero.','Final repartition worsened.','Pooled policy requested after first resplit result; no actual early stop or candidate change.'],'training':'Reuse exact full500 CNN seed checkpoints024; train missing24 nestedCNNs via133 node; refit4affines and48 residual models including none. Existing all500 independent KRR/S/B/shortB retained.','test_scope':'Infer only after training validation; no test fit/selection/calibration/clipping.','prior_inspection':'During cache inspection historical004 report displayed prior test aggregates. They did not affect the already-fixed recipe.','holdout_scope':'All500 now authorized for training; no formerholdout subgroup scoring.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':previous.check()['protected'],'cost':0,'uploaded':False};e.write(_repo_path(OUT, 'decision.json'),p);e.emit({'candidate015_locked':True,'new_nested_CNN_fits':24,'reused_full500_CNNs':3})
def check():
 previous.check();x.check();p=e.read(_repo_path(OUT, 'decision.json'))
 for name,h in {**p['hashes'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, name))==h,name
 return p
def inputs(train=False):
 d,P,cells=previous.inputs();P=P.copy();new=[];root=_repo_path(OUT, 'cnn-nested')
 if train:root.mkdir(exist_ok=False)
 for j,(A,B,M,D,short) in enumerate(cells):
  M=M.copy();D=D.copy();M[:,2]=x.node(d,A,B,100*(j+1),M[:,2],root/f'middle-{j}',train);P[B,2]=M[:,2];lookup={int(i):k for k,i in enumerate(A)}
  for ell in range(3):
   C,E,DP=a.bundle(c.OUT/f'block-{j}/deep-{ell}');assert set(C)|set(E)==set(A) and not(set(C)|set(E))&set(B)
   D[[lookup[int(i)] for i in E],2]=x.node(d,C,E,100*(j+1)+100*(ell+1),DP[:,2],root/f'deep-{j}-{ell}',train)
  new.append((A,B,M,D,short));e.emit({'full500_nested_block':j,'training':train})
 return d,P,new

def fit():
 check();start=time.perf_counter();d,P,cells=inputs(train=True);ip=a.mixture(P,ARM);_,cal=a.affine(ip,d['y'],ip);e.write(_repo_path(OUT, 'calibrator.json'),{**cal,'n_fit':500,'train_ids':d['ids']})
 with (_repo_path(OUT, 'calibration_training.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),components=P,inner_oof=ip)
 records=[]
 for j,(A,B,M,D,short) in enumerate(cells):
  path=_repo_path(OUT, f'block-{j}');path.mkdir(exist_ok=False);pred,coef=a.affine(a.mixture(D,ARM),d['y'][A],a.mixture(M,ARM));anchor=.9*pred+.1*short;cor,rec=a.fit_block(d,B,anchor,0,j,path/'residual')
  with (path/'anchor.npz').open('xb') as f:np.savez_compressed(_repo_io(f),anchor_indices=A,block_indices=B,deep_components=D,middle_components=M,short=short,anchor=anchor)
  record={'block':j,'calibrator':coef,'anchor_rows':len(A),'residual_rows':len(B),'chosen':rec['chosen'],'anchor_sha256':e.sha(path/'anchor.npz'),'residual_report_sha256':e.sha(path/'residual/report.json')};e.write(path/'report.json',record);records.append(record);e.emit({'full500_block':j,'chosen':rec['chosen']})
 report={'n_train':500,'train_ids':d['ids'],'new_affine_fits':4,'new_residual_blocks':3,'new_residual_models_including_none':48,'new_neural_fits':24,'reused_full500_CNNs':3,'remaining_base_models_already_trained_on500':True,'train_csv_sha256':e.sha(_repo_path(ROOT, 'data/train.csv')),'calibrator_sha256':e.sha(_repo_path(OUT, 'calibrator.json')),'calibration_training_sha256':e.sha(_repo_path(OUT, 'calibration_training.npz')),'blocks':records,'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'seconds':time.perf_counter()-start,'holdout_subgroup_scored':False,'test_decoded':False};e.write(_repo_path(OUT, 'fit.json'),report);check();e.emit({'full500_downstream_fit_complete':True,'seconds':report['seconds']})

def inference(device='mps'):
 check();v=e.read(_repo_path(OUT, 'training-validation.json'));assert v['passed'] and v['fit_sha256']==e.sha(_repo_path(OUT, 'fit.json'));ids,_=c.cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);m8=e.read(_repo_path(ROOT, 'artifacts/submission-candidate-008.manifest.json'));model8=e.read(p8.OUT/'report.json');features={};heads={}
 for arm in ['S','B']:
  path=_repo_path(ROOT, m8[f'test_{arm}_cache']);assert e.sha(path)==m8[f'test_{arm}_sha256'];z=a.readnp(path);assert z['ids'].tolist()==ids;features[arm]=z['X'].astype(float);pp=[]
  for rec in model8['models']:
   if rec['arm']==arm:pp.append(p8.predict_checkpoint(p8.OUT/rec['checkpoint'],features[arm]))
  heads[arm]=np.mean(pp,axis=0)
 z=a.readnp(_repo_path(ROOT, 'artifacts/final-model-053/verification-test-compact-cnn-inputs.npz'));assert z['ids'].tolist()==ids;kernel=joblib.load(_repo_path(ROOT, 'artifacts/final-model-017/kernel.joblib'));kp=kernel['model'].predict(z['compact']);cnn=np.mean([c.cp.predict_cnn(z['cnn_images'],cnnpath(seed),device) for seed in x.SEEDS],axis=0);raw=np.column_stack([np.zeros(1000),kp,cnn,heads['S'],heads['B']]);mixture=a.mixture(raw,ARM);cal=e.read(_repo_path(OUT, 'calibrator.json'));calpred=cal['target_center']+cal['slope']*(mixture-cal['prediction_center']);saved=torch.load(_repo_path(ROOT, 'artifacts/final-model-086/head/model.pt'),weights_only=True,map_location='cpu');small=e.Small(saved['scaler_mean'].numpy(),saved['scaler_scale'].numpy(),20270215);small.load_state_dict({k:saved['state'][k] for k in small.state_dict()});small=small.to(device);short=e.small_predict(small,features['B'],np.arange(1000),device=device)*saved['target_std']+saved['target_mean'];del small;torch.mps.empty_cache();base=.9*calpred+.1*short
 z=a.readnp(c.OUT/'test57.npz',e.read(c.OUT/'test57.json')['sha256']);assert z['ids'].tolist()==ids;X=z['X'];cor=[]
 for j in range(3):
  path=_repo_path(OUT, f'block-{j}/residual');rec=e.read(path/'report.json');assert e.sha(path/'selected.joblib')==rec['models_sha256']['selected.joblib'];cor.append(n.predict(joblib.load(_repo_io(path/'selected.joblib'))['model'],X))
 cor=np.stack(cor);pred=base+cor.mean(0);assert pred.shape==(1000,) and np.isfinite(pred).all();return ids,raw,short,base,cor,pred,X

def predict():
 ids,raw,short,base,cor,pred,X=inference();assert len(set(ids))==1000
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','kernel','cnn','Smean','Bmean','shortB','new_base','residual0','residual1','residual2','candidate015']);w.writerows(zip(ids,*raw[:,1:].T,short,base,*cor,pred))
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),components=raw,short=short,base=base,corrections=cor,primary=pred)
 fit=e.read(_repo_path(OUT, 'fit.json'));lo,hi=e.read(p8.OUT/'report.json')['training_label_range'];manifest={'candidate':'015','rows':1000,'columns':['ID','hardness'],'submission_sha256':e.sha(CSV),'components_sha256':e.sha(CSV.with_suffix('.components.csv')),'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'training_validation_sha256':e.sha(_repo_path(OUT, 'training-validation.json')),'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'sample_submission_sha256':e.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'selected':[v['chosen'] for v in fit['blocks']],'prediction_range':[float(pred.min()),float(pred.max())],'training_label_range':[lo,hi],'outside_training_range':int(((pred<lo)|(pred>hi)).sum()),'finite':True,'duplicates':0,'ID_order_exact':True,'inference':'Fresh full500 KRR, three identity CNN seeds equally averaged, S/B heads, shortB, and new calibrator/residuals; no795baseline.','clipping':False,'test_fit_or_tuning':False,'holdout_subgroup_scored':False,'014_preserved':True,'uploaded':False,'cost':0};e.write(CSV.with_suffix('.manifest.json'),manifest);check();e.emit({'CSV_ready':True,'SHA256':e.sha(CSV),'range':manifest['prediction_range']})
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit','predict']);args=p.parse_args();globals()[args.mode]()
