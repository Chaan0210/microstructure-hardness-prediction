"""Locked080 adapter on all500, retaining009 affine coefficients and all other models."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,csv,datetime,argparse,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_adapter_080 as study
import candidate_pipeline as cp
import candidate006_pipeline as imagecode
import numpy as np,torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
fv=study.fv;OUT=_repo_path(ROOT, 'artifacts/final-model-081');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-010.csv');CFG=study.CFG

def prepare():
 plan=study.check();v=json.loads((_repo_path(ROOT, 'reports/dinob-adapter-080-validation.json')).read_text());assert v['passed'] and v['eligible_for_candidate'] and len(v['results'])==3 and all(r['comparison']['gain']>0 for r in v['results'].values());OUT.mkdir(exist_ok=False)
 decision={'candidate':'010','locked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formula':'candidate009 + preserved009_slope*.125*(adapted_B_mean3 - frozen_B_mean3)','recipe':'Same080 rank4 adapters on finalB qkv/proj; sameCFG8fixedepochs, all500rows, seed20270201. Retain alreadyall500trained threeBheads, scalers and originalbackbone, S/legacy models and009 affine coefficients unchanged. No new fit or selection of calibration/weights.','config':CFG,'validation_sha256':fv.sha(_repo_path(ROOT, 'reports/dinob-adapter-080-validation.json')),'development_results':v['results'],'authorization':'User authorized candidate preparation after consistent development improvement. Formerholdout includedonlyafterlock forfull500fit; no subgroup evaluation. Testonlyaftermodelcomplete. Uploadreservedforparent.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['candidate010_pipeline.py','verify_candidate010.py','dinob_adapter_080.py','validate_dinob_adapter_080.py','candidate006_pipeline.py','candidate_pipeline.py']},'protected':plan['protected'],'base_assets':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/final-model-065/report.json','artifacts/final-model-075/calibrator.json','artifacts/submission-candidate-009.csv','artifacts/submission-candidate-008.components.csv','artifacts/submission-candidate-008.manifest.json','reports/candidate008-verification.json','reports/candidate009-verification.json']},'cost':0,'uploaded':False,'limits':v['limits']+' Gains are only0.0008..0.0014RMSE and eachconditionalCIincludeszero; this is an exploratory candidate, not a reliable Public improvement forecast.'};fv.write(_repo_path(OUT, 'decision.json'),decision);print(json.dumps({'locked':True,'candidate':'010','formula':decision['formula']}),flush=True)

def check():
 study.check();d=json.loads((_repo_path(OUT, 'decision.json')).read_text());assert d['candidate']=='010' and d['config']==CFG and fv.sha(_repo_path(ROOT, 'reports/dinob-adapter-080-validation.json'))==d['validation_sha256']
 for n,s in {**d['sources'],**d['protected'],**d['base_assets']}.items():assert fv.sha(_repo_path(ROOT, n))==s,n
 return d

def load_training():
 check();r=json.loads((_repo_path(ROOT, 'artifacts/final-model-065/report.json')).read_text());ids,y=cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);assert r['train_ids']==ids and r['train_csv_sha256']==fv.sha(_repo_path(ROOT, 'data/train.csv'));cache=_repo_path(ROOT, r['train_B_cache']);assert fv.sha(cache)==r['train_B_sha256']
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids;B=z['X'].astype(float)
 sc=StandardScaler().fit(B);states=[];pred=[];headsha={}
 for seed in study.SEEDS:
  rec=next(a for a in r['models'] if a['arm']=='B' and a['seed']==seed);p=_repo_path(ROOT, 'artifacts/final-model-065', rec['checkpoint']);assert fv.sha(p)==rec['checkpoint_sha256'];a=torch.load(_repo_io(p),weights_only=True,map_location='cpu');assert a['train_ids']==ids and a['seed']==seed and a['config']==study.h.CONFIG and a['effective_aux_weight']==0.;assert a['target_mean']==float(y.mean()) and a['target_std']==float(y.std())
  for n,val in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(a[n].numpy(),val)
  h=study.h.Head(6912);h.load_state_dict(a['model']);pred.append(study.h.predict(h,sc.transform(B),a['target_mean'],a['target_std']));states.append(a);headsha[str(p.relative_to(ROOT))]=fv.sha(p)
 return {'ids':ids,'y':y,'B':B,'states':states,'frozenB':np.mean(pred,0),'source_report':r,'head_sha256':headsha}

def cache():
 check();torch.set_num_threads(2);d=load_training();m=study.backbone.model().to('mps');before=study.h.state_hash(m.state_dict());path=_repo_path(OUT, 'train_prefix.npy');assert not path.exists();a=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(500,1297,768));start=time.perf_counter();hashes={}
 for i,ident in enumerate(d['ids']):
  raw=_repo_path(ROOT, 'data/train', f'{ident}.png');hashes[ident]=fv.sha(raw);x=torch.tensor(imagecode.input_image(raw)[None],device='mps');z=study.prefix(m,x);a[i]=z.cpu().numpy()[0]
  with torch.no_grad():
   zz=m.norm(m.blocks[-1](z));p=zz[:,1:];f=torch.cat((zz[:,0],p.mean(1),p.std(1,correction=0)),1).cpu().numpy()[0]
  np.testing.assert_array_equal(f,d['B'][i,-2304:])
  if (i+1)%100==0:print(json.dumps({'stage':'full500_prefix','rows':i+1,'seconds':time.perf_counter()-start}),flush=True)
 a.flush();del a;assert hashes==d['source_report']['training_image_sha256'] and before==study.h.state_hash(m.state_dict());fv.write(_repo_path(OUT, 'prefix.json'),{'ids':d['ids'],'shape':[500,1297,768],'sha256':fv.sha(path),'training_image_sha256':hashes,'frozen500_descriptor_exact':True,'seconds':time.perf_counter()-start})

def get_cache(d):
 r=json.loads((_repo_path(OUT, 'prefix.json')).read_text());assert r['ids']==d['ids'] and fv.sha(_repo_path(OUT, 'train_prefix.npy'))==r['sha256'];return np.load(_repo_path(OUT, 'train_prefix.npy'),mmap_mode='r')

def train_adapter(d,tokens,path):
 path.mkdir(exist_ok=False);m=study.backbone.model();t=study.Tail(m,d['states'],CFG['seed']).to('mps');initial=study.h.state_hash(t.state_dict());frozen=study.frozen_hash(t);ym=float(d['y'].mean());ys=float(d['y'].std());params=[p for p in t.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=CFG['lr'],weight_decay=CFG['weight_decay']);orders=np.random.default_rng(CFG['seed']).permuted(np.tile(np.arange(500),(CFG['epochs'],1)),axis=1);curve=[];start=time.perf_counter()
 for epoch,order in enumerate(orders):
  losses=[]
  for first in range(0,500,CFG['effective_batch']):
   group=order[first:first+CFG['effective_batch']];opt.zero_grad(set_to_none=True)
   for offset in range(0,len(group),CFG['microbatch']):
    ii=group[offset:offset+CFG['microbatch']];p=t(study.tensor(tokens[ii]),study.tensor(d['B'][ii,:4608]));loss=F.smooth_l1_loss(p,study.tensor((d['y'][ii]-ym)/ys),beta=1.)+CFG['anchor_weight']*F.mse_loss(p,study.tensor((d['frozenB'][ii]-ym)/ys));assert torch.isfinite(loss);(loss*(len(ii)/len(group))).backward();losses.append(float(loss.detach()))
   gn=torch.nn.utils.clip_grad_norm_(params,CFG['clip']);assert torch.isfinite(gn);opt.step()
  curve.append({'epoch':epoch+1,'train_objective':float(np.mean(losses)),'seconds':time.perf_counter()-start});print(json.dumps({'stage':'full500_adapter',**curve[-1]}),flush=True)
 assert study.frozen_hash(t)==frozen and all(p.grad is None for p in m.parameters()) and all(not a.training for a in t.modules());pred=study.predict(t,d,tokens,np.arange(500))*ys+ym;state={n:v.detach().cpu() for n,v in t.state_dict().items()};torch.save({'state':state,'train_ids':d['ids'],'n_train':500,'config':CFG,'target_mean':ym,'target_std':ys,'head_sha256':d['head_sha256'],'initial_state_sha256':initial,'frozen_state_sha256':frozen},_repo_io(path/'model.pt'))
 with (path/'training_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),frozenB=d['frozenB'],adaptedB=pred)
 r={'n_train':500,'train_ids':d['ids'],'config':CFG,'curve':curve,'initial_state_sha256':initial,'frozen_state_sha256':frozen,'final_state_sha256':study.h.state_hash(state),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'checkpoint_sha256':fv.sha(path/'model.pt'),'training_predictions_sha256':fv.sha(path/'training_predictions.npz'),'head_sha256':d['head_sha256'],'seconds':time.perf_counter()-start,'no_holdout_subgroup_metrics':True};fv.write(path/'report.json',r);return r

def fit():
 check();torch.set_num_threads(2);d=load_training();tokens=get_cache(d);r=train_adapter(d,tokens,_repo_path(OUT, 'adapter'));cal=_repo_path(ROOT, 'artifacts/final-model-075/calibrator.json');report={'n_train':500,'train_ids':d['ids'],'adapter_report_sha256':fv.sha(_repo_path(OUT, 'adapter/report.json')),'checkpoint_sha256':r['checkpoint_sha256'],'training_image_sha256':d['source_report']['training_image_sha256'],'train_csv_sha256':fv.sha(_repo_path(ROOT, 'data/train.csv')),'unchanged009_calibrator_sha256':fv.sha(cal),'prefix_sha256':fv.sha(_repo_path(OUT, 'train_prefix.npy')),'head_sha256':d['head_sha256'],'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json')),'no_test_access':True,'no_holdout_score':True};fv.write(_repo_path(OUT, 'report.json'),report);check();print(json.dumps({'fit_complete':True,'n_train':500,'seconds':r['seconds']}),flush=True)

def test_inputs():
 check();assert (_repo_path(OUT, 'report.json')).exists();ids,_=cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);man=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-008.manifest.json')).read_text());p=_repo_path(ROOT, man['test_B_cache']);assert fv.sha(p)==man['test_B_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:assert z['ids'].tolist()==ids;B=z['X'].astype(float)
 assert fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-008.components.csv'))==man['components_sha256']
 with (_repo_path(ROOT, 'artifacts/submission-candidate-008.components.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;oldB=np.array([float(r['B_mean3']) for r in rows])
 with (_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;old9=np.array([float(r['hardness']) for r in rows]);return ids,B,oldB,old9,man

def infer_all(ids,B,states,checkpoint):
 m=study.backbone.model().to('mps');before=study.h.state_hash(m.state_dict());t=study.Tail(m,states,CFG['seed']).to('mps');t.load_state_dict(checkpoint['state']);pred=[];hashes={};start=time.perf_counter()
 with torch.no_grad():
  for j,ident in enumerate(ids):
   path=_repo_path(ROOT, 'data/test', f'{ident}.png');x=torch.tensor(imagecode.input_image(path)[None],device='mps');z=study.prefix(m,x);p=t(z,study.tensor(B[j:j+1,:4608])).cpu().numpy().astype(float)*checkpoint['target_std']+checkpoint['target_mean'];pred.append(float(p[0]));hashes[ident]=fv.sha(path)
   if (j+1)%200==0:print(json.dumps({'stage':'full1000_inference','rows':j+1,'seconds':time.perf_counter()-start}),flush=True)
 assert before==study.h.state_hash(m.state_dict());return np.array(pred),hashes

def predict():
 check();torch.set_num_threads(2);fit=json.loads((_repo_path(OUT, 'report.json')).read_text());assert fv.sha(_repo_path(OUT, 'adapter/model.pt'))==fit['checkpoint_sha256'];d=load_training();a=torch.load(_repo_path(OUT, 'adapter/model.pt'),weights_only=True,map_location='cpu');ids,B,oldB,old9,man=test_inputs();base=[]
 for state in d['states']:
  m=study.h.Head(6912);m.load_state_dict(state['model']);base.append(study.h.predict(m,(B-state['scaler_mean'].numpy())/state['scaler_scale'].numpy(),state['target_mean'],state['target_std']))
 np.testing.assert_array_equal(np.mean(base,0),oldB);newB,hashes=infer_all(ids,B,d['states'],a);assert hashes==man['test_image_sha256'];cal=json.loads((_repo_path(ROOT, 'artifacts/final-model-075/calibrator.json')).read_text());p=old9+cal['slope']*.125*(newB-oldB);assert p.shape==(1000,) and np.isfinite(p).all() and len(set(ids))==1000
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,p))
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','candidate009','frozenB','adaptedB','candidate010']);w.writerows(zip(ids,old9,oldB,newB,p))
 with (_repo_path(OUT, 'test_predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),candidate009=old9,frozenB=oldB,adaptedB=newB,primary=p)
 manifest={'candidate':'010','rows':1000,'columns':['ID','hardness'],'finite':True,'exact_sample_order':True,'duplicates':0,'submission_sha256':fv.sha(CSV),'components_sha256':fv.sha(CSV.with_suffix('.components.csv')),'model_report_sha256':fv.sha(_repo_path(OUT, 'report.json')),'checkpoint_sha256':fv.sha(_repo_path(OUT, 'adapter/model.pt')),'predictions_sha256':fv.sha(_repo_path(OUT, 'test_predictions.npz')),'sample_submission_sha256':fv.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'test_image_sha256':hashes,'formula':'candidate009+unchanged009slope*.125*(adaptedB-frozenB)','unchanged009_slope':cal['slope'],'unchanged009_calibrator_sha256':fv.sha(_repo_path(ROOT, 'artifacts/final-model-075/calibrator.json')),'best009_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')),'prediction_range':[float(p.min()),float(p.max())],'difference_from009_RMS':float(np.sqrt(np.mean((p-old9)**2))),'test_fit':False,'clipping':False,'holdout_scored':False,'uploaded':False,'cost':0};fv.write(CSV.with_suffix('.manifest.json'),manifest);check();print(json.dumps({k:v for k,v in manifest.items() if k!='test_image_sha256'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','cache','fit','predict']);a=ap.parse_args();{'prepare':prepare,'cache':cache,'fit':fit,'predict':predict}[a.mode]()
