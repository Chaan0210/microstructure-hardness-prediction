"""Fixed TabICL v2 development-only study. No holdout/test file access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
from pathlib import Path
P=Path(__file__).resolve().parent
R=_RepoPath(__file__).resolve().parents[3]
for k,v in {'HF_HOME':_repo_path(P, 'hf-cache'),'TORCH_HOME':_repo_path(P, 'torch-cache'),'TMPDIR':_repo_path(P, 'tmp'),'XDG_CACHE_HOME':_repo_path(P, 'cache')}.items(): os.environ[k]=str(v)
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, hashlib, time, resource, sys, gc, importlib.metadata
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from tabicl import TabICLRegressor
import joblib
threadpool_limits(2); torch.set_num_threads(2)
CK=_repo_path(P, 'downloads/tabicl-regressor-v2-20260212.ckpt')
PARAMS=dict(n_estimators=1,batch_size=1,kv_cache=False,use_amp=False,n_jobs=2,device='cpu',model_path=str(CK),allow_auto_download=False,random_state=42,use_fa3=False,disk_offload_dir=str(_repo_path(P, 'offload')))
ARMS=['compact26','dinob_pca64','compact26_dinob_pca64']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f: json.dump(x,f,indent=2,allow_nan=False)
def rmse(y,p): return float(np.sqrt(np.mean((y-p)**2)))
def inputs(split):
 paths=[_repo_path(R, 'artifacts/run-012/development_features.npz'),_repo_path(R, 'artifacts/dinob-062/development_features.npz'),_repo_path(R, f'artifacts/bootstrap-selector-144/{split}/predictions.npz')]
 c,b,ref=[np.load(_repo_io(f),allow_pickle=False) for f in paths]
 assert sha(paths[1])==json.loads((_repo_path(R, 'artifacts/dinob-062/extraction.json')).read_text())['feature_sha256']
 assert sha(paths[2])==json.loads(paths[2].with_name('report.json').read_text())['predictions_sha256']
 ids=ref['ids']; assert len(set(ids))==400 and len(ids)==400
 def align(d): return np.array([{str(x):i for i,x in enumerate(d['ids'])}[str(x)] for x in ids])
 ci,bi=align(c),align(b)
 assert np.array_equal(c['y'][ci],ref['y']) and len(set(c['groups']))==400
 assert np.isfinite(c['X']).all() and np.isfinite(b['X']).all()
 return ids,c['X'][ci],b['X'][bi],ref['y'],ref['fold'],ref['primary'],{str(f.relative_to(R)):sha(f) for f in paths}
def features(arm,C,B,t,v):
 if arm=='compact26': return C[t],C[v],None
 ss=StandardScaler().fit(B[t]); pca=PCA(n_components=64,svd_solver='full').fit(ss.transform(B[t]))
 xt=pca.transform(ss.transform(B[t])); xv=pca.transform(ss.transform(B[v]))
 if arm=='compact26_dinob_pca64': xt=np.column_stack([C[t],xt]); xv=np.column_stack([C[v],xv])
 return xt,xv,{'scaler':ss,'pca':pca}
def fit_one(arm,split,fold,benchmark=False):
 ids,C,B,y,folds,ref,source=inputs(split); t=np.where(folds!=fold)[0];v=np.where(folds==fold)[0]
 out=_repo_path(P, 'results', split, arm, str(fold));out.mkdir(parents=True,exist_ok=True)
 if (out/'report.json').exists(): return json.loads((out/'report.json').read_text())
 assert not (out/'prediction.npz').exists(), 'Partial prior attempt; inspect before resume'
 print('START',split,arm,fold,len(t),len(v),flush=True)
 start=time.perf_counter(); xt,xv,transform=features(arm,C,B,t,v)
 preprocessing_seconds=time.perf_counter()-start
 model=TabICLRegressor(**PARAMS); a=time.perf_counter();model.fit(xt,y[t]);fit_seconds=time.perf_counter()-a
 a=time.perf_counter(); pred=model.predict(xv);predict_seconds=time.perf_counter()-a
 assert pred.shape==(len(v),) and np.isfinite(pred).all()
 model.save(_repo_io(out/'model.pkl'),save_model_weights=False,save_training_data=True,save_kv_cache=False)
 if transform is not None: joblib.dump(transform,_repo_io(out/'transform.joblib'))
 actual_configs={str(k):len(v) for k,v in model.ensemble_generator_.ensemble_configs_.items()}
 replay={}
 if benchmark:
  a=time.perf_counter(); repeat=model.predict(xv);replay['repeat_seconds']=time.perf_counter()-a
  replay['repeat_max_abs']=float(np.max(np.abs(pred-repeat)))
  del model;gc.collect()
  loaded=TabICLRegressor.load(out/'model.pkl',device='cpu');reloaded=loaded.predict(xv)
  replay['reload_max_abs']=float(np.max(np.abs(pred-reloaded)))
  assert loaded.allow_auto_download is False
  del loaded;gc.collect()
  fresh=TabICLRegressor(**PARAMS).fit(xt,y[t]);replay['fresh_fit_max_abs']=float(np.max(np.abs(pred-fresh.predict(xv))))
  assert max(v for k,v in replay.items() if k.endswith('max_abs'))==0
  del fresh
 np.savez_compressed(_repo_io(out/'prediction.npz'),ids=ids[v],y=y[v],prediction=pred,reference018=ref[v],fold=fold)
 report={'split':split,'arm':arm,'fold':int(fold),'train_ids':ids[t].tolist(),'validation_ids':ids[v].tolist(),'train_shape':list(xt.shape),'validation_shape':list(xv.shape),'preprocessing_seconds':preprocessing_seconds,'fit_seconds':fit_seconds,'predict_seconds':predict_seconds,'total_seconds':time.perf_counter()-start,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'finite':True,'rmse':rmse(y[v],pred),'reference018_rmse':rmse(y[v],ref[v]),'fixed20blend_rmse':rmse(y[v],.8*ref[v]+.2*pred),'reproducibility':replay,'actual_ensemble_configs':actual_configs,'source_sha256':source,'prediction_sha256':sha(out/'prediction.npz'),'model_sha256':sha(out/'model.pkl')}
 write(out/'report.json',report);print('DONE',json.dumps({k:v for k,v in report.items() if not k.endswith('_ids') and k!='source_sha256'}),flush=True)
 gc.collect();return report

def summarize(split,arm):
 ids,C,B,y,folds,ref,source=inputs(split);pred=np.full(len(y),np.nan)
 for fold in sorted(set(folds)):
  f=_repo_path(P, 'results', split, arm, str(fold), 'prediction.npz');d=np.load(_repo_io(f),allow_pickle=False);v=np.where(folds==fold)[0]
  assert np.array_equal(ids[v],d['ids']);pred[v]=d['prediction']
 assert np.isfinite(pred).all();blend=.8*ref+.2*pred
 scores={'standalone':rmse(y,pred),'fixed20blend':rmse(y,blend),'reference018':rmse(y,ref),'fold_blend_gains':[rmse(y[folds==f],ref[folds==f])-rmse(y[folds==f],blend[folds==f]) for f in sorted(set(folds))]}
 scores['qualifies_resplits']=scores['fixed20blend']<scores['reference018'] and sum(x>0 for x in scores['fold_blend_gains'])>=3
 scores['residual_correlation']=float(np.corrcoef(y-ref,y-pred)[0,1])
 out=_repo_path(P, 'results', split, arm)
 np.savez_compressed(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=folds,reference018=ref,prediction=pred,fixed20blend=blend)
 write(out/'summary.json',scores); print('SUMMARY',split,arm,json.dumps(scores),flush=True);return scores

def main():
 mode=sys.argv[1]
 if mode=='plan':
  _,_,_,_,folds,_,sources=inputs('original')
  plan={'study':162,'arms':ARMS,'dino':'official frozen ViT-B6912 existing development cache; StandardScaler then full-SVD PCA64, fitted on fold train only','parameters':PARAMS,'fixed_blend':'0.8 * saved best018 outer OOF + 0.2 * TabICL outer OOF; no learned weights','resplit_gate':'original fixed20blend improves overall RMSE and >=3/5 folds; only qualifying arms evaluated on existing 20261031 and 20261101 folds','continuation_gate':'same fixed20blend improves all three split RMSE; otherwise stop, no full500/test','benchmark':'original compact26 lowest fold, repeat predict, save/load, independent fresh fit; CPU peak RSS is process lifetime maximum','holdout_test':'not read; no selection on either','source_sha256':sources,'versions':{x:importlib.metadata.version(x) for x in ['tabicl','torch','numpy','scipy','scikit-learn']},'fold_values':[int(x) for x in sorted(set(folds))],'limitations':'many adaptive studies on same 400 IDs; resplits not independent samples; TabICL in-context inference rather than gradient training'}
  write(_repo_path(P, 'plan.json'),plan);print(json.dumps(plan,indent=2))
 elif mode=='benchmark': fit_one('compact26','original',int(min(inputs('original')[4])),True)
 elif mode=='original':
  for arm in ARMS:
   for f in sorted(set(inputs('original')[4])): fit_one(arm,'original',int(f))
   summarize('original',arm)
 elif mode=='resplits':
  arms=[a for a in ARMS if json.loads((_repo_path(P, 'results/original', a, 'summary.json')).read_text())['qualifies_resplits']]
  for split in ['20261031','20261101']:
   for arm in arms:
    for f in sorted(set(inputs(split)[4])): fit_one(arm,split,int(f))
    summarize(split,arm)
  print('QUALIFYING_ARMS',arms,flush=True)
 else: raise ValueError(mode)
if __name__=='__main__': main()
