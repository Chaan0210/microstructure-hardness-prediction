"""Locked exploratory frozen-head complement; train500 then infer1000."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,csv,datetime,argparse,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import frozen_complement_085 as study
import candidate_pipeline as cp
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
e=study.e;fv=study.fv;OUT=_repo_path(ROOT, 'artifacts/final-model-086');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-011.csv');CFG=e.CFG

def prepare():
 plan=study.check();v=json.loads((_repo_path(ROOT, 'reports/frozen-complement-085-validation.json')).read_text());assert v['passed'] and v['eligible_for_candidate'] and len(v['results'])==3;OUT.mkdir(exist_ok=False)
 d={'candidate':'011','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config':CFG,'formula':'.9*candidate009+.1*fresh_frozen_B_head','full500_rule':plan['full500_rule'],'selection_bias':plan['selection_bias'],'limits':plan['limits'],'development_results':v['results'],'validation_sha256':fv.sha(_repo_path(ROOT, 'reports/frozen-complement-085-validation.json')),'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['candidate011_pipeline.py','verify_candidate011.py','frozen_complement_085.py','dinob_joint_084.py']},'base_assets':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/final-model-065/report.json','artifacts/final-model-075/calibrator.json','artifacts/submission-candidate-009.csv','artifacts/submission-candidate-008.manifest.json','reports/candidate009-verification.json','reports/candidate008-verification.json']},'authorization':'User authorized full500 fit and candidate011 CSV verification after both unchanged resplits improve. No holdout subgroup evaluation.010 unsubmitted. No upload.','cost':0};fv.write(_repo_path(OUT, 'decision.json'),d);print(json.dumps({'locked':True,'candidate':'011','formula':d['formula']}),flush=True)

def check():
 study.check();d=json.loads((_repo_path(OUT, 'decision.json')).read_text());assert d['config']==CFG and d['candidate']=='011' and fv.sha(_repo_path(ROOT, 'reports/frozen-complement-085-validation.json'))==d['validation_sha256']
 for n,s in {**d['sources'],**d['base_assets']}.items():assert fv.sha(_repo_path(ROOT, n))==s,n
 return d

def training():
 check();r=json.loads((_repo_path(ROOT, 'artifacts/final-model-065/report.json')).read_text());ids,y=cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);assert r['train_ids']==ids and r['train_csv_sha256']==fv.sha(_repo_path(ROOT, 'data/train.csv'));p=_repo_path(ROOT, r['train_B_cache']);assert fv.sha(p)==r['train_B_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:assert z['ids'].tolist()==ids;B=z['X'].astype(float)
 assert B.shape==(500,6912) and np.isfinite(B).all();return {'ids':ids,'y':y,'B':B,'source_report':r}

def train(d,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(2);sc=StandardScaler().fit(d['B']);ym=float(d['y'].mean());ys=float(d['y'].std());m=e.backbone.model();model=e.Model(m,sc,CFG['seed'],False).to('mps');initial=e.h.state_hash(model.state_dict());fixed=e.fixed_hash(model);opt=e.optimizer(model);orders=np.random.default_rng(CFG['seed']).permuted(np.tile(np.arange(500),(12,1)),axis=1);curve=[];seeds=[]
 for ep,order in enumerate(orders):
  e.schedule(opt,ep);model.training_mode();losses=[]
  for first in range(0,500,16):
   group=order[first:first+16];opt.zero_grad(set_to_none=True)
   for offset in range(0,len(group),2):
    ii=group[offset:offset+2];seed=e.seed_for(0,ep,first+offset);torch.manual_seed(seed);seeds.append(seed);p=model(e.tensor(d['B'][ii]));loss=F.smooth_l1_loss(p,e.tensor((d['y'][ii]-ym)/ys),beta=1.);assert torch.isfinite(loss);(loss*(len(ii)/len(group))).backward();losses.append(float(loss.detach()))
   gn=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.);assert torch.isfinite(gn);opt.step()
  tp=e.predict(model,d,None,np.arange(500))*ys+ym;rec={'epoch':ep+1,'online_train_loss':float(np.mean(losses)),'train_RMSE':fv.rmse(d['y'],tp),'head_lr':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-start};curve.append(rec);print(json.dumps({'stage':'full500_frozen',**rec}),flush=True)
 assert e.fixed_hash(model)==fixed and all(p.grad is None for p in model.block.parameters());state={n:v.detach().cpu() for n,v in model.state_dict().items()};torch.save({'state':state,'train_ids':d['ids'],'n_train':500,'config':CFG,'target_mean':ym,'target_std':ys,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'initial_state_sha256':initial,'fixed_state_sha256':fixed},_repo_io(path/'model.pt'))
 with (path/'training_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),predictions=tp)
 r={'n_train':500,'train_ids':d['ids'],'config':CFG,'curve':curve,'initial_state_sha256':initial,'fixed_state_sha256':fixed,'final_state_sha256':e.h.state_hash(state),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_seeds_sha256':hashlib.sha256(np.array(seeds,dtype=np.int64).tobytes()).hexdigest(),'checkpoint_sha256':fv.sha(path/'model.pt'),'predictions_sha256':fv.sha(path/'training_predictions.npz'),'no_holdout_subgroup_metrics':True,'seconds':time.perf_counter()-start};fv.write(path/'report.json',r);return r

def fit():
 check();d=training();r=train(d,_repo_path(OUT, 'head'));base=d['source_report'];report={'n_train':500,'train_ids':d['ids'],'head_report_sha256':fv.sha(_repo_path(OUT, 'head/report.json')),'checkpoint_sha256':r['checkpoint_sha256'],'train_csv_sha256':fv.sha(_repo_path(ROOT, 'data/train.csv')),'training_image_sha256':base['training_image_sha256'],'train_B_cache':base['train_B_cache'],'train_B_sha256':base['train_B_sha256'],'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json')),'unchanged009_calibrator_sha256':fv.sha(_repo_path(ROOT, 'artifacts/final-model-075/calibrator.json')),'no_test_access':True,'no_holdout_score':True};fv.write(_repo_path(OUT, 'report.json'),report);check();print(json.dumps({'fit_complete':True,'seconds':r['seconds']}),flush=True)

def test_inputs():
 check();assert (_repo_path(OUT, 'report.json')).exists();ids,_=cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);man=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-008.manifest.json')).read_text());p=_repo_path(ROOT, man['test_B_cache']);assert fv.sha(p)==man['test_B_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:assert z['ids'].tolist()==ids;B=z['X'].astype(float)
 with (_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;old=np.array([float(r['hardness']) for r in rows]);assert B.shape==(1000,6912) and np.isfinite(B).all();return ids,B,old,man

def infer(B,a,device='mps'):
 sc=StandardScaler();sc.mean_=a['scaler_mean'].numpy();sc.scale_=a['scaler_scale'].numpy();m=e.Model(e.backbone.model(),sc,CFG['seed'],False);m.load_state_dict(a['state']);m=m.to(device).eval();pred=[]
 with torch.no_grad():
  for first in range(0,len(B),2):pred.append(m(torch.tensor(B[first:first+2],dtype=torch.float32,device=device)).cpu().numpy().astype(float)*a['target_std']+a['target_mean'])
 return np.concatenate(pred)

def predict():
 check();torch.set_num_threads(2);r=json.loads((_repo_path(OUT, 'report.json')).read_text());assert fv.sha(_repo_path(OUT, 'head/model.pt'))==r['checkpoint_sha256'];a=torch.load(_repo_path(OUT, 'head/model.pt'),weights_only=True,map_location='cpu');ids,B,old,man=test_inputs();new=infer(B,a);p=.9*old+.1*new;assert len(set(ids))==1000 and p.shape==(1000,) and np.isfinite(p).all()
 for path,columns,values in [(CSV,['ID','hardness'],zip(ids,p)),(CSV.with_suffix('.components.csv'),['ID','candidate009','frozen_head','candidate011'],zip(ids,old,new,p))]:
  with path.open('x',newline='') as f:w=csv.writer(f);w.writerow(columns);w.writerows(values)
 with (_repo_path(OUT, 'test_predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),candidate009=old,frozen_head=new,primary=p)
 manifest={'candidate':'011','rows':1000,'columns':['ID','hardness'],'finite':True,'exact_sample_order':True,'duplicates':0,'submission_sha256':fv.sha(CSV),'components_sha256':fv.sha(CSV.with_suffix('.components.csv')),'model_report_sha256':fv.sha(_repo_path(OUT, 'report.json')),'checkpoint_sha256':fv.sha(_repo_path(OUT, 'head/model.pt')),'predictions_sha256':fv.sha(_repo_path(OUT, 'test_predictions.npz')),'sample_submission_sha256':fv.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'test_B_cache':man['test_B_cache'],'test_B_sha256':man['test_B_sha256'],'test_image_sha256':man['test_image_sha256'],'formula':'.9*009+.1*fresh_frozen_B_head','unchanged009_calibrator_sha256':r['unchanged009_calibrator_sha256'],'protected009_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')),'protected010_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-010.csv')),'prediction_range':[float(p.min()),float(p.max())],'difference_from009_RMS':float(np.sqrt(np.mean((p-old)**2))),'test_fit':False,'clipping':False,'holdout_scored':False,'uploaded':False,'cost':0};fv.write(CSV.with_suffix('.manifest.json'),manifest);check();print(json.dumps({k:v for k,v in manifest.items() if k!='test_image_sha256'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','fit','predict']);a=ap.parse_args();{'prepare':prepare,'fit':fit,'predict':predict}[a.mode]()
