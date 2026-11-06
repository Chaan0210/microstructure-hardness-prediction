"""Private, offline, bounded v2 in-context regression. No raw-image or final-model work."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,time,json,hashlib,socket,subprocess,re,resource,threading,gc,argparse
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];assert Path.cwd().resolve()==R and Path(sys.prefix).resolve()==_repo_path(R, '.venv') and not (_repo_path(R, '.env')).exists()
network_events=[]
def blocked(*args,**kwargs):network_events.append('blocked');raise RuntimeError('Offline run: network/auth/model download prohibited')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.getaddrinfo=blocked

def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if p.is_relative_to(_repo_path(P, 'data')):raise RuntimeError('All rawdata/holdout/test files prohibited')
  if (p.name in ['.env','.netrc','.git-credentials','credentials','token','stored_tokens'] or any(v in p.parts for v in ['.aws','.ssh'])) and not p.is_relative_to(R):raise RuntimeError('Outside credential/dotenv access prohibited')
sys.addaudithook(guard)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as fp:
  for b in iter(lambda:fp.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as fp:json.dump(z,fp,indent=2,allow_nan=False)
def emit(z):print(json.dumps(z),flush=True)
def swap():
 out=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',out).groups();return int(float(v)*(1024**2 if u=='M' else 1024**3))
def check():
 p=read(_repo_path(R, 'plan.json'));a=read(_repo_path(R, 'authorization-and-preservation.json'))
 for n,h in {**p['input_sha256'],**p['protected'],**p['original_environment_metadata_sha256']}.items():assert sha(_repo_path(P, n))==h,n
 site=_repo_path(R, '.venv/lib/python3.13/site-packages')
 for n,h in p['official_source_sha256'].items():assert sha(site/n)==h,n
 for n,h in read(_repo_path(R, 'code-lock.json'))['sources'].items():assert sha(_repo_path(R, n))==h,n
 assert sha(_repo_path(R, 'weights/tabpfn-v2-regressor.ckpt'))==a['model_sha256'];return p

class Monitor:
 def __init__(self,stage,rep):
  self.stage=stage;self.rep=rep;self.samples=[];self.stop=threading.Event();self.started=time.monotonic();self.initialswap=swap()
  if not (_repo_path(R, 'budget-start.json')).exists():write(_repo_path(R, 'budget-start.json'),{'start_monotonic':self.started,'initial_swap':self.initialswap})
  self.total=read(_repo_path(R, 'budget-start.json'))
  path=_repo_path(R, rep+'-budget-start.json')
  if not path.exists():write(path,{'start_monotonic':self.started})
  self.repstart=read(path)['start_monotonic']
 def sample(self):
  now=time.monotonic();rss=int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024;s=swap();z={'stage_seconds':now-self.started,'total_seconds':now-self.total['start_monotonic'],'representation_seconds':now-self.repstart,'RSS':rss,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'swap':s};self.samples.append(z);reasons=[]
  if z['peak_RSS']>=6000000000:reasons.append('RSS6GB')
  if s>self.total['initial_swap']:reasons.append('swap_growth')
  if z['total_seconds']>=1800:reasons.append('total30min')
  if z['representation_seconds']>=600:reasons.append('representation10min')
  if reasons:write(_repo_path(R, self.stage+'-STOP.json'),{'reasons':reasons,'trace':self.samples});emit({'STOP':reasons,'resources':z});os._exit(74)
 def loop(self):
  while not self.stop.wait(1):
   try:self.sample()
   except Exception as e:write(_repo_path(R, self.stage+'-monitor-error.json'),{'error':repr(e),'trace':self.samples});os._exit(75)
 def __enter__(self):self.sample();self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start();return self
 def __exit__(self,*args):self.stop.set();self.worker.join();self.sample();write(_repo_path(R, self.stage+'-resources.json'),{'trace':self.samples,'stage_seconds':time.monotonic()-self.started})

def main(mode):
 plan=check();rep='compact26' if mode=='benchmark' else mode
 if mode!='benchmark':assert read(_repo_path(R, 'benchmark.json'))['passed']
 with Monitor(mode,rep) as mon:
  import numpy as np,torch,joblib
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  from threadpoolctl import threadpool_limits
  threadpool_limits(1);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
  from tabpfn import TabPFNRegressor,ModelSpecs
  from tabpfn import model_loading
  model_loading.download_model=blocked
  from tabpfn.settings import settings
  assert settings.tabpfn.model_version.value=='v2' and Path(settings.tabpfn.model_cache_dir).resolve()==_repo_path(R, 'cache/tabpfn')
  weight=_repo_path(R, 'weights/tabpfn-v2-regressor.ckpt');models,criterion,configs,ic=model_loading.load_model_criterion_config(model_path=weight,version='v2',estimator_type='regressor',check_bar_distribution_criterion=True,download_if_not_exists=False,devices=[torch.device('cpu')],force_inference_dtype=torch.float32)
  assert len(models)==1 and 'v2' in type(configs[0]).__module__ and 'v2_5' not in type(configs[0]).__module__
  model=models[0];model.requires_grad_(False);model.eval();assert all(v.device.type=='cpu' and v.dtype==torch.float32 for v in model.parameters())
  def statehash():
   h=hashlib.sha256()
   for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
   return h.hexdigest()
  initial=statehash();spec=ModelSpecs(model=model,architecture_config=configs[0],inference_config=ic,norm_criterion=criterion)
  if mode=='benchmark':
   missing=_repo_path(R, 'weights/absent-v2-regressor.ckpt')
   try:model_loading.load_model_criterion_config(model_path=missing,version='v2',estimator_type='regressor',check_bar_distribution_criterion=True,download_if_not_exists=False)
   except ValueError as e:assert 'downloading is disabled' in str(e)
   else:raise AssertionError('Missingmodeldidnotfailclosed')
   write(_repo_path(R, 'loader-validation.json'),{'passed':True,'explicit_v2':True,'architecture':type(configs[0]).__module__+'.'+type(configs[0]).__name__,'criterion':type(criterion).__name__,'ModelSpecs_constructor':True,'all_parameters_CPU_float32':True,'missing_file_failed_before_download':True,'network_attempts':len(network_events),'foundation_state_sha256':initial})
  def load(path):
   with np.load(_repo_path(P, path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
  c=load('artifacts/run-012/development_features.npz');d=load('experiments/dinol-177/features.npz');g=load('experiments/dinol-177/original/predictions.npz');assert read(_repo_path(P, 'experiments/dinol-177/original/validation.json'))['passed']
  for a in [d,g]:np.testing.assert_array_equal(c['ids'],a['ids'])
  np.testing.assert_array_equal(c['y'],g['y']);np.testing.assert_array_equal(c['groups'],g['g']);np.testing.assert_array_equal(g['blend_L_MLP'],.9*g['reference018']+.1*g['L_MLP']);assert c['X'].shape==(400,26) and d['X'].shape==(400,9216)
  y=c['y'];fold=g['fold'];groups=g['g'];root=_repo_path(R, 'results', rep);root.mkdir(parents=True,exist_ok=True);folds=[0] if mode=='benchmark' else range(1 if rep=='compact26' else 0,5);records=[]
  for k in folds:
   start=time.monotonic();T=np.flatnonzero(fold!=k);V=np.flatnonzero(fold==k);assert len(T)==320 and len(V)==80 and not set(T)&set(V) and not set(groups[T])&set(groups[V]);out=root/f'fold-{k}';out.mkdir(exist_ok=False)
   if rep=='compact26':Xtr=c['X'][T].astype(np.float32);Xva=c['X'][V].astype(np.float32)
   else:
    pre=_repo_path(R, 'PCA', f'fold-{k}.joblib');pre.parent.mkdir(exist_ok=True)
    if rep=='DINO_L_PCA64':
     X=d['X'][T].astype(float);sc=StandardScaler().fit(X);Z=sc.transform(X);pca=PCA(n_components=64,svd_solver='full',whiten=False).fit(Z);joblib.dump({'scaler':sc,'PCA':pca,'train_indices':T,'validation_indices':V},_repo_io(pre))
    obj=joblib.load(_repo_io(pre));sc=obj['scaler'];pca=obj['PCA'];np.testing.assert_array_equal(obj['train_indices'],T);np.testing.assert_array_equal(obj['validation_indices'],V);X=d['X'][T].astype(float);np.testing.assert_allclose(sc.mean_,X.mean(0),atol=1e-12,rtol=0);np.testing.assert_allclose(sc.var_,X.var(0),atol=1e-12,rtol=0);assert sc.n_samples_seen_==pca.n_samples_==320;assert pca.components_.shape==(64,9216);np.testing.assert_allclose(pca.mean_,sc.transform(X).mean(0),atol=1e-12)
    Ztr=pca.transform(sc.transform(X));Zva=pca.transform(sc.transform(d['X'][V].astype(float)));manual=(sc.transform(d['X'][V].astype(float))-pca.mean_)@pca.components_.T;np.testing.assert_allclose(Zva,manual,atol=1e-11)
    Xtr=(Ztr if rep=='DINO_L_PCA64' else np.column_stack([c['X'][T],Ztr])).astype(np.float32);Xva=(Zva if rep=='DINO_L_PCA64' else np.column_stack([c['X'][V],Zva])).astype(np.float32)
   assert np.isfinite(Xtr).all() and np.isfinite(Xva).all();seed=202710184+k
   est=TabPFNRegressor(model_path=spec,n_estimators=1,auto_scale_n_estimators=False,device='cpu',inference_precision=torch.float32,fit_mode='low_memory',memory_saving_mode=True,n_preprocessing_jobs=1,random_state=seed,show_progress_bar=False)
   fs=time.monotonic();est.fit(Xtr,y[T]);fit_seconds=time.monotonic()-fs;ps=time.monotonic();pred=est.predict(Xva);predict_seconds=time.monotonic()-ps;assert pred.shape==(80,) and np.isfinite(pred).all();assert len(est.executor_.ensemble_preprocessor.configs)==1;assert est.executor_.X_train.shape[0]==320;assert all(str(v)=='cpu' for v in est.devices_);assert statehash()==initial
   file=out/'estimator.joblib';joblib.dump(est,_repo_io(file),compress=0);del est;gc.collect();again=joblib.load(_repo_io(file));ps=time.monotonic();reload_pred=again.predict(Xva);reload_seconds=time.monotonic()-ps;np.testing.assert_array_equal(pred,reload_pred);assert len(network_events)==0
   with (out/'predictions.npz').open('xb') as fp:np.savez_compressed(_repo_io(fp),train_indices=T,validation_indices=V,ids=c['ids'][V],prediction=pred)
   report={'fold':k,'representation':rep,'seed':seed,'shape_train':list(Xtr.shape),'shape_validation':list(Xva.shape),'fit_seconds':fit_seconds,'predict_seconds':predict_seconds,'reload_predict_seconds':reload_seconds,'fold_total_seconds':time.monotonic()-start,'train_validation_groups_disjoint':True,'all_preprocessing_train_only':True,'reload_exact':True,'foundation_unchanged':True,'n_estimators':1,'network_attempts':len(network_events),'standalone_RMSE':float(np.sqrt(np.mean((y[V]-pred)**2))),'best021_RMSE':float(np.sqrt(np.mean((y[V]-g['blend_L_MLP'][V])**2))),'blend05_RMSE':float(np.sqrt(np.mean((y[V]-(.95*g['blend_L_MLP'][V]+.05*pred))**2))),'model_sha256':sha(file),'predictions_sha256':sha(out/'predictions.npz')};write(out/'report.json',report);records.append(report);emit(report);del again;gc.collect()
  if mode=='benchmark':
   elapsed=time.monotonic()-mon.started;passed=elapsed<120 and 5*elapsed<600 and max(z['peak_RSS'] for z in mon.samples)<6000000000 and swap()==mon.total['initial_swap'];write(_repo_path(R, 'benchmark.json'),{'passed':bool(passed),'seconds_including_import_load_fit_save_reload':elapsed,'projected5fold_seconds_conservative':5*elapsed,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'swap_growth':swap()-mon.total['initial_swap'],'finite':True,'first_fold_report':records[0],'noholdout_test':True});emit({'benchmark_passed':bool(passed),'seconds':elapsed})
  else:
   pred=np.full(400,np.nan);rr=[]
   for k in range(5):
    path=root/f'fold-{k}';rec=read(path/'report.json');rr.append(rec);assert sha(path/'predictions.npz')==rec['predictions_sha256']
    with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:V=z['validation_indices'];np.testing.assert_array_equal(z['ids'],c['ids'][V]);pred[V]=z['prediction']
   assert np.isfinite(pred).all();ref=g['blend_L_MLP'];blend=.95*ref+.05*pred;rmse=lambda p:float(np.sqrt(np.mean((y-p)**2)));fg=[float(np.sqrt(np.mean((y[fold==k]-ref[fold==k])**2))-np.sqrt(np.mean((y[fold==k]-blend[fold==k])**2))) for k in range(5)]
   with (root/'OOF.npz').open('xb') as fp:np.savez_compressed(_repo_io(fp),ids=c['ids'],y=y,groups=groups,fold=fold,standalone=pred,best021=ref,blend05=blend)
   report={'representation':rep,'RMSE':{'best021':rmse(ref),'standalone':rmse(pred),'blend05':rmse(blend)},'blend05_gain':rmse(ref)-rmse(blend),'fold_blend_gains':fg,'positive_folds':sum(v>0 for v in fg),'residual_correlation_vs021':float(np.corrcoef(y-pred,y-ref)[0,1]),'blend_update_RMS':float(np.sqrt(np.mean((blend-ref)**2))),'all5_reload_exact':True,'all5_train320_validation80':True,'CPU_threads':torch.get_num_threads(),'fold_report_sha256':{str(k):sha(root/f'fold-{k}/report.json') for k in range(5)},'OOF_sha256':sha(root/'OOF.npz'),'network_attempts':len(network_events),'all5_foundation_weights_unchanged':True,'holdout_test_used':False,'private_benchmark_not_submission':True};write(root/'report.json',report);emit(report)
  check()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','compact26','DINO_L_PCA64','joint90']);a=p.parse_args()
 try:main(a.mode)
 except Exception as e:
  write(_repo_path(R, a.mode+'-failure.json'),{'error_type':type(e).__name__,'error':str(e),'network_attempts':len(network_events)});raise
