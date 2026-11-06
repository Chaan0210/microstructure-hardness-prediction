"""Frozen137 quadratic extension of015; full500 downstream-only refit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,csv,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate015_pipeline as previous
import quadratic_calibration_137 as q
import numpy as np,torch,joblib
x=previous.x;a=previous.a;e=previous.e;n=previous.n;r=previous.r;c=previous.c;p8=previous.p8;vn=previous.vn;ARM=previous.ARM
OUT=_repo_path(ROOT, 'artifacts/final-model-138');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-016.csv')
cnnpath=previous.cnnpath
inputs=previous.inputs

def prepare():
 previous.check();q.check();assert not CSV.exists();OUT.mkdir(exist_ok=False);decision=e.read(q.OUT/'decision.json');assert decision['eligible'] and decision['independent_validation_passed'];names=['candidate016_pipeline.py','validate_candidate016.py','quadratic_calibration_137.py','validate_quadratic_calibration_137.py','reports/quadratic-calibration-137-summary.json','artifacts/quadratic-calibration-137/decision.json','reports/candidate015-public-submission.json','artifacts/submission-candidate-015.csv','artifacts/submission-candidate-015.manifest.json','artifacts/final-model-134/fit.json','artifacts/final-model-134/training-validation.json']
 for node in (previous.OUT/'cnn-nested').rglob('*'):
  if node.is_file():names.append(str(node.relative_to(ROOT)))
 p={'candidate':'016','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formula':'same015 fixedraw mixture; .9*train-only quadratic137 calibration(raw)+.1shortB+mean3newhonestphysics57residuals','lambda':1.,'extra_curvature_df_per_calibrator':.5,'development_decision':decision,'training':'Exactly137:4 quadratic calibrators and48 residual models includingnone. Reuse verifiedfull500 independentbaselearners andall133/134 3seednestedCNNmeans. No newneuralfits.','holdout':'Included onlyinlockedfinal500fit; no subgroup score orselection.','test':'Onlyafterfit andindependenttrainingvalidation; no fit/tuning/clipping.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':q.check()['protected'],'best015_retained':True,'cost':0,'uploaded':False};e.write(_repo_path(OUT, 'decision.json'),p);e.emit({'candidate016_locked':True,'new_neural_fits':0,'new_quadratic_calibrators':4})
def check():
 previous.check();q.check();p=e.read(_repo_path(OUT, 'decision.json'))
 for f,h in {**p['hashes'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 return p

def fit():
 check();start=time.perf_counter();d,P,cells=inputs();ip=a.mixture(P,ARM);cal=q.fit(ip,d['y']);e.write(_repo_path(OUT, 'calibrator.json'),{**cal,'n_fit':500,'train_ids':d['ids']})
 with (_repo_path(OUT, 'calibration_training.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),components=P,inner_oof=ip)
 records=[]
 for j,(A,B,M,D,short) in enumerate(cells):
  path=_repo_path(OUT, f'block-{j}');path.mkdir(exist_ok=False);coef=q.fit(a.mixture(D,ARM),d['y'][A]);pred=q.predict(coef,a.mixture(M,ARM));anchor=.9*pred+.1*short;cor,rec=a.fit_block(d,B,anchor,0,j,path/'residual')
  with (path/'anchor.npz').open('xb') as f:np.savez_compressed(_repo_io(f),anchor_indices=A,block_indices=B,deep_components=D,middle_components=M,short=short,anchor=anchor)
  record={'block':j,'calibrator':coef,'anchor_rows':len(A),'residual_rows':len(B),'chosen':rec['chosen'],'anchor_sha256':e.sha(path/'anchor.npz'),'residual_report_sha256':e.sha(path/'residual/report.json')};e.write(path/'report.json',record);records.append(record);e.emit({'full500_block':j,'chosen':rec['chosen']})
 report={'n_train':500,'train_ids':d['ids'],'new_quadratic_calibrators':4,'new_residual_blocks':3,'new_residual_models_including_none':48,'new_neural_fits':0,'reused_full500_CNNs':3,'remaining_base_models_already_trained_on500':True,'train_csv_sha256':e.sha(_repo_path(ROOT, 'data/train.csv')),'calibrator_sha256':e.sha(_repo_path(OUT, 'calibrator.json')),'calibration_training_sha256':e.sha(_repo_path(OUT, 'calibration_training.npz')),'blocks':records,'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'seconds':time.perf_counter()-start,'holdout_subgroup_scored':False,'test_decoded':False};e.write(_repo_path(OUT, 'fit.json'),report);check();e.emit({'full500_downstream_fit_complete':True,'seconds':report['seconds']})

def inference(device='mps'):
 check();v=e.read(_repo_path(OUT, 'training-validation.json'));assert v['passed'] and v['fit_sha256']==e.sha(_repo_path(OUT, 'fit.json'));ids,_=c.cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);m8=e.read(_repo_path(ROOT, 'artifacts/submission-candidate-008.manifest.json'));model8=e.read(p8.OUT/'report.json');features={};heads={}
 for arm in ['S','B']:
  path=_repo_path(ROOT, m8[f'test_{arm}_cache']);assert e.sha(path)==m8[f'test_{arm}_sha256'];z=a.readnp(path);assert z['ids'].tolist()==ids;features[arm]=z['X'].astype(float);pp=[]
  for rec in model8['models']:
   if rec['arm']==arm:pp.append(p8.predict_checkpoint(p8.OUT/rec['checkpoint'],features[arm]))
  heads[arm]=np.mean(pp,axis=0)
 z=a.readnp(_repo_path(ROOT, 'artifacts/final-model-053/verification-test-compact-cnn-inputs.npz'));assert z['ids'].tolist()==ids;kernel=joblib.load(_repo_path(ROOT, 'artifacts/final-model-017/kernel.joblib'));kp=kernel['model'].predict(z['compact']);cnn=np.mean([c.cp.predict_cnn(z['cnn_images'],cnnpath(seed),device) for seed in x.SEEDS],axis=0);raw=np.column_stack([np.zeros(1000),kp,cnn,heads['S'],heads['B']]);mixture=a.mixture(raw,ARM);cal=e.read(_repo_path(OUT, 'calibrator.json'));calpred=q.predict(cal,mixture);saved=torch.load(_repo_path(ROOT, 'artifacts/final-model-086/head/model.pt'),weights_only=True,map_location='cpu');small=e.Small(saved['scaler_mean'].numpy(),saved['scaler_scale'].numpy(),20270215);small.load_state_dict({k:saved['state'][k] for k in small.state_dict()});small=small.to(device);short=e.small_predict(small,features['B'],np.arange(1000),device=device)*saved['target_std']+saved['target_mean'];del small;torch.mps.empty_cache();base=.9*calpred+.1*short
 z=a.readnp(c.OUT/'test57.npz',e.read(c.OUT/'test57.json')['sha256']);assert z['ids'].tolist()==ids;X=z['X'];cor=[]
 for j in range(3):
  path=_repo_path(OUT, f'block-{j}/residual');rec=e.read(path/'report.json');assert e.sha(path/'selected.joblib')==rec['models_sha256']['selected.joblib'];cor.append(n.predict(joblib.load(_repo_io(path/'selected.joblib'))['model'],X))
 cor=np.stack(cor);pred=base+cor.mean(0);assert pred.shape==(1000,) and np.isfinite(pred).all();return ids,raw,short,base,cor,pred,X

def predict():
 ids,raw,short,base,cor,pred,X=inference();assert len(set(ids))==1000
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','kernel','cnn','Smean','Bmean','shortB','new_base','residual0','residual1','residual2','candidate016']);w.writerows(zip(ids,*raw[:,1:].T,short,base,*cor,pred))
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),components=raw,short=short,base=base,corrections=cor,primary=pred)
 fit=e.read(_repo_path(OUT, 'fit.json'));lo,hi=e.read(p8.OUT/'report.json')['training_label_range'];manifest={'candidate':'016','rows':1000,'columns':['ID','hardness'],'submission_sha256':e.sha(CSV),'components_sha256':e.sha(CSV.with_suffix('.components.csv')),'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'training_validation_sha256':e.sha(_repo_path(OUT, 'training-validation.json')),'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'sample_submission_sha256':e.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'selected':[v['chosen'] for v in fit['blocks']],'prediction_range':[float(pred.min()),float(pred.max())],'training_label_range':[lo,hi],'outside_training_range':int(((pred<lo)|(pred>hi)).sum()),'finite':True,'duplicates':0,'ID_order_exact':True,'inference':'Fresh full500 KRR, three identity CNN seeds equally averaged, S/B heads, shortB, and new calibrator/residuals; no795baseline.','clipping':False,'test_fit_or_tuning':False,'holdout_subgroup_scored':False,'015_preserved':True,'uploaded':False,'cost':0};e.write(CSV.with_suffix('.manifest.json'),manifest);check();e.emit({'CSV_ready':True,'SHA256':e.sha(CSV),'range':manifest['prediction_range']})
if __name__=='__main__':
 torch.set_num_threads(2);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','fit','predict']);args=parser.parse_args();globals()[args.mode]()
