"""Locked nested affine calibration; all500 calibration fit, unchanged full500008."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv,time,argparse,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nested_calibration_074 as study
import candidate_pipeline as cp
import numpy as np,torch,joblib
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
fv=study.fv;OUT=_repo_path(ROOT, 'artifacts/final-model-075');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')

def prepare():
 p=study.check();v=json.loads((_repo_path(ROOT, 'reports/nested-calibration-074-validation.json')).read_text());assert v['passed'] and v['eligible_for_candidate'] and len(v['results'])==3 and all(r['comparison']['gain']>0 for r in v['results'].values());pub=json.loads((_repo_path(ROOT, 'reports/candidate008-public-submission.json')).read_text());OUT.mkdir(exist_ok=False)
 decision={'candidate':'009','locked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formula':'target_center+slope*(candidate008-prediction_center)','recipe':'Generatefull500train-onlyOOF with3foldGroupKFoldseed20270110andentire008pipeline,thenfitoneOLSaffinecalibrator onall500pairs. Unchangedunderlying008modelsalreadytrainedall500areretainedverbatim. No model/weight/strength selection.',
 'authorization':'Delegateduserauthorizedtrainingandverificationofimprovementcandidates;074passesall3developmentpartitions. Uploadreservedforparent.','model_scope':'Newcalibratorfitwithall500labels. Underlyingfull500008modelsnotneedlesslyretrained;innerOOFmodelsfreshfit333/334trainingrows,notusedasfinalinferenceensemble.',
 'no_holdout_score':'Former100 includedonlyinlocked500fit;nosubgroupmetrics/selection. Full500OOFisusedtofitcalibrator,notpresentedasnewvalidationevidence.',
 'test':'Aftermodelcomplete,useunchangedverified008testpredictionsasfixedbaseinferencecache;applynewcalibrator. No testfit/tuning/clipping/selection.',
 'validation_sha256':fv.sha(_repo_path(ROOT, 'reports/nested-calibration-074-validation.json')),'results':v['results'],'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['candidate009_pipeline.py','verify_candidate009.py','nested_calibration_074.py','candidate_pipeline.py','candidate_features.py','validate_nested_calibration_074_v2.py']},'protected':{**p['protected'],**pub['protected_sha256'],'reports/nested-calibration-074-validator-repair.json':fv.sha(_repo_path(ROOT, 'reports/nested-calibration-074-validator-repair.json'))},'best008_sha256':pub['sha256'],'cost':0,'new_packages':0,'uploaded':False,'limits':v['limits']};fv.write(_repo_path(OUT, 'decision.json'),decision);print(json.dumps({'locked':True,'candidate':'009','formula':decision['formula']}),flush=True)

def check():
 study.check();d=json.loads((_repo_path(OUT, 'decision.json')).read_text());assert d['candidate']=='009';assert fv.sha(_repo_path(ROOT, 'reports/nested-calibration-074-validation.json'))==d['validation_sha256']
 for path,sha in {**d['sources'],**d['protected']}.items():assert fv.sha(_repo_path(ROOT, path))==sha,path
 return d

def load_training():
 report=json.loads((_repo_path(OUT, 'report.json')).read_text());assert fv.sha(_repo_path(OUT, 'training-inputs.npz'))==report['training_inputs_sha256']
 with np.load(_repo_path(OUT, 'training-inputs.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 d['ids']=d['ids'].tolist();bundle=joblib.load(_repo_path(ROOT, 'artifacts/run-005/best_model.joblib'));d['recipes']={n:clone(m) for n,m in bundle['models'].items()};d['weights']=bundle['weights'];d['blocks']=json.loads((_repo_path(ROOT, 'artifacts/final-model-017/report.json')).read_text())['blocks'];return d

def fit():
 decision=check();torch.set_num_threads(2);start=time.perf_counter();ids,y=cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);base8=json.loads((_repo_path(ROOT, 'artifacts/final-model-065/report.json')).read_text());assert base8['train_ids']==ids and base8['train_csv_sha256']==fv.sha(_repo_path(ROOT, 'data/train.csv'));d={'ids':ids,'y':y}
 for arm in ('S','B'):
  path=_repo_path(ROOT, base8[f'train_{arm}_cache']);assert fv.sha(path)==base8[f'train_{arm}_sha256']
  with np.load(_repo_io(path),allow_pickle=False) as z:assert z['ids'].tolist()==ids;d[arm]=z['X'].astype(float)
 with np.load(_repo_path(ROOT, 'artifacts/run-003/train_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);d['Xbase']=z['X'].copy()
 raw3=json.loads((_repo_path(ROOT, 'artifacts/final-model-017/report.json')).read_text());assert raw3['training_id_order']==ids;d['K'],d['images'],hashes=cp.extract_all(ids,'train','run-012',raw3['feature_names']);assert hashes==base8['training_image_sha256']==raw3['training_image_sha256'];d['blocks']=raw3['blocks'];sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());d['g']=np.array([sp['group_by_id'][i] for i in ids]);bundle=joblib.load(_repo_path(ROOT, 'artifacts/run-005/best_model.joblib'));d['recipes']={n:clone(m) for n,m in bundle['models'].items()};d['weights']=bundle['weights']
 with np.load(_repo_io(study.OUT/'development_features.npz'),allow_pickle=False) as z:
  ix=[ids.index(i) for i in z['ids'].tolist()];np.testing.assert_array_equal(d['Xbase'][ix],z['Xbase']);np.testing.assert_array_equal(d['images'][ix],z['images'])
 with (_repo_path(OUT, 'training-inputs.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),**{n:np.array(d[n]) for n in ['ids','y','g','S','B','Xbase','K','images']})
 folds=list(GroupKFold(3,shuffle=True,random_state=study.INNER_SEED).split(d['S'],y,d['g']));ip=np.full(500,np.nan);fid=np.full(500,-1,int)
 for j,(tr,va) in enumerate(folds):
  ip[va],_=study.fit_cell(d,tr,va,np.array([],dtype=int),0,j,_repo_path(OUT, f'inner-{j}'));fid[va]=j
 center=float(ip.mean());targetcenter=float(y.mean());pc=ip-center;slope=float(pc@(y-targetcenter)/(pc@pc));assert np.isfinite(ip).all() and np.isfinite(slope)
 with (_repo_path(OUT, 'calibration_training.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),fold=fid,inner_oof=ip)
 calibrator={'prediction_center':center,'target_center':targetcenter,'slope':slope,'intercept':targetcenter-slope*center,'n_fit':500,'train_ids':ids,'training_predictions_sha256':fv.sha(_repo_path(OUT, 'calibration_training.npz')),'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json')),'no_hyperparameter_selection':True};fv.write(_repo_path(OUT, 'calibrator.json'),calibrator);report={'n_train':500,'baseline_source_cache_sha256':fv.sha(_repo_path(ROOT, 'artifacts/run-003/train_features.npz')),'training_inputs_sha256':fv.sha(_repo_path(OUT, 'training-inputs.npz')),'calibrator_sha256':fv.sha(_repo_path(OUT, 'calibrator.json')),'train_csv_sha256':fv.sha(_repo_path(ROOT, 'data/train.csv')),'training_image_sha256':hashes,'inner_report_sha256':{f'inner-{j}/report.json':fv.sha(_repo_path(OUT, f'inner-{j}/report.json')) for j in range(3)},'unchanged_base008':True,'base008_models_trained_on500':True,'calibrator_fit_on500':True,'no_holdout_subgroup_score':True,'no_test_access':True,'seconds':time.perf_counter()-start};check();fv.write(_repo_path(OUT, 'report.json'),report);print(json.dumps({'fit_complete':True,'n_train':500,'slope':slope,'seconds':report['seconds']}),flush=True)

def predict():
 d=check();fit=json.loads((_repo_path(OUT, 'report.json')).read_text());assert fv.sha(_repo_path(OUT, 'calibrator.json'))==fit['calibrator_sha256'];c=json.loads((_repo_path(OUT, 'calibrator.json')).read_text());ids,_=cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);v8=json.loads((_repo_path(ROOT, 'reports/candidate008-verification.json')).read_text());assert v8['valid'] and v8['submission_sha256']==d['best008_sha256'];source=_repo_path(ROOT, 'artifacts/submission-candidate-008.csv');assert fv.sha(source)==d['best008_sha256']
 with source.open(newline='') as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;base=np.array([float(r['hardness']) for r in rows]);p=c['target_center']+c['slope']*(base-c['prediction_center']);assert p.shape==(1000,) and np.isfinite(p).all() and len(set(ids))==1000
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,p))
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','candidate008','calibrated009']);w.writerows(zip(ids,base,p))
 manifest={'candidate':'009','rows':1000,'columns':['ID','hardness'],'finite':True,'duplicate_ids':0,'exact_sample_order':True,'submission_sha256':fv.sha(CSV),'components_sha256':fv.sha(CSV.with_suffix('.components.csv')),'sample_submission_sha256':fv.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'model_report_sha256':fv.sha(_repo_path(OUT, 'report.json')),'calibrator_sha256':fv.sha(_repo_path(OUT, 'calibrator.json')),'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json')),'base008_sha256':d['best008_sha256'],'base_inference_source':'Immutable008full1000inferencecache,previouslyverifiedfromfull500models;notusedforselection.','prediction_range':[float(p.min()),float(p.max())],'formula':d['formula'],'slope':c['slope'],'prediction_center':c['prediction_center'],'target_center':c['target_center'],'clipping':False,'test_fit':False,'holdout_scored':False,'uploaded':False,'cost':0};fv.write(CSV.with_suffix('.manifest.json'),manifest);check();print(json.dumps(manifest),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','fit','predict']);a=ap.parse_args();{'prepare':prepare,'fit':fit,'predict':predict}[a.mode]()
