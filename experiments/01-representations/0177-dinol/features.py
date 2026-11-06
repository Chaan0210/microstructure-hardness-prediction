"""Approved177 frozenL feature benchmark/extraction; development400 only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
os.environ['XFORMERS_DISABLED']='1'
import sys,time,json,hashlib,resource,socket,subprocess,re,threading,argparse
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];V=_repo_path(P, 'vendor/dinov2-official/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8');sys.path.insert(0,str(V))
def blocked(*a,**kw):raise RuntimeError('Network disabled after authorized download')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np,torch
from PIL import Image
from threadpoolctl import threadpool_limits
threadpool_limits(2);torch.set_num_threads(2)
from dinov2.hub.backbones import dinov2_vitl14

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def emit(r):print(json.dumps(r),flush=True)
def check():
 p=read(_repo_path(R, 'plan.json'));v=read(_repo_path(R, 'provenance.json'));assert sha(_repo_path(R, 'dinov2_vitl14_pretrain.pth'))==v['sha256']
 for f,h in p['official_code'].items():assert sha(_repo_path(V, f))==h
 for f,h in read(_repo_path(R, 'feature-code-lock.json'))['sources'].items():assert sha(_repo_path(R, f))==h
 return p

def swap():
 x=subprocess.check_output(['sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',x).groups();return int(float(v)*(1024**2 if u=='M' else 1024**3))
def resources():
 rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024;driver=torch.mps.driver_allocated_memory();out=subprocess.check_output(['memory_pressure','-Q'],text=True);free=int(re.search(r'free percentage: (\d+)%',out).group(1));return {'RSS':rss,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver':driver,'conservative_RSS_plus_MPS':rss+driver,'swap_bytes':swap(),'system_memory_free_percent':free}
class Monitor:
 def __init__(self,stage,limits):self.stage=stage;self.limits=limits;self.trace=[];self.stop=threading.Event();self.start=time.perf_counter();self.initial=swap()
 def sample(self):
  z={'seconds':time.perf_counter()-self.start,**resources()};self.trace.append(z);lim=self.limits;fail=[]
  if z['peak_RSS']>lim['max_RSS_bytes'] or z['MPS_driver']>lim['max_MPS_driver_bytes'] or z['conservative_RSS_plus_MPS']>lim['max_conservative_current_RSS_plus_MPS_bytes']:fail.append('memory_budget_6GB')
  if z['swap_bytes']-self.initial>lim['swap_growth_stop_bytes']:fail.append('swap_growth')
  if z['system_memory_free_percent']<lim['minimum_memory_free_percent']:fail.append('system_memory_pressure')
  deadline=lim['max_benchmark_seconds'] if self.stage=='benchmark' else lim['max_feature400_seconds']
  if z['seconds']>deadline:fail.append('time_budget')
  if fail:write(_repo_path(R, f'{self.stage}-resource-stop.json'),{'stopped':True,'reasons':fail,'trace':self.trace});emit({'STOP':fail,'resources':z});os._exit(70)
 def loop(self):
  while not self.stop.wait(2):
   try:self.sample()
   except Exception as e:write(_repo_path(R, f'{self.stage}-monitor-error.json'),{'error':repr(e),'trace':self.trace});os._exit(71)
 def __enter__(self):self.sample();self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start();return self
 def __exit__(self,*args):self.stop.set();self.worker.join();self.sample();write(_repo_path(R, f'{self.stage}-resources.json'),{'trace':self.trace,'initial_swap_bytes':self.initial,'seconds':time.perf_counter()-self.start,'limits':self.limits})

def scope_guard(plan):
 allowed={_repo_path(P, 'data/train', f'{i}.png') for i in plan['IDs']}
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   f=Path(os.fsdecode(args[0])).absolute()
   if f.is_relative_to(_repo_path(P, 'data')) and f not in allowed:raise RuntimeError('Holdout/test/rawCSV access blocked: '+str(f))
 sys.addaudithook(audit)
def image_input(i):
 with Image.open(_repo_path(P, 'data/train', f'{i}.png')) as im:
  assert im.mode=='L' and im.size==(256,256);a=np.asarray(im.resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/255.
 return np.ascontiguousarray((np.repeat(a[None],3,axis=0)-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None])
def model():
 m=dinov2_vitl14(pretrained=False);sd=torch.load(_repo_path(R, 'dinov2_vitl14_pretrain.pth'),map_location='cpu',weights_only=True);m.load_state_dict(sd,strict=True);del sd;m.requires_grad_(False);m.eval();assert len(m.blocks)==24 and m.embed_dim==1024 and m.num_register_tokens==0;return m

def statehash(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def feature(m,x):
 with torch.inference_mode():
  layers=m.get_intermediate_layers(x,n=[11,17,23],return_class_token=True,norm=True);assert all(p.shape==(1,1296,1024) and c.shape==(1,1024) for p,c in layers);z=torch.cat([torch.cat([c,p.mean(1),p.std(1,correction=0)],1) for p,c in layers],1).cpu().numpy()[0]
 assert z.shape==(9216,) and np.isfinite(z).all();return z

def benchmark():
 p=check();scope_guard(p);start=time.perf_counter();result={};outputs={}
 with Monitor('benchmark',p['resources']) as mon:
  m=model();initial=statehash(m);params=sum(v.numel() for v in m.parameters());xx={i:torch.from_numpy(image_input(i)[None]) for i in p['smoke_IDs']}
  for device in ['cpu','mps']:
   m=m.to(device);feature(m,xx[p['smoke_IDs'][0]].to(device));times=[];out=[]
   for i in p['smoke_IDs']:
    st=time.perf_counter();a=feature(m,xx[i].to(device));times.append(time.perf_counter()-st);out.append(a);mon.sample()
   repeated=feature(m,xx[p['smoke_IDs'][0]].to(device));np.testing.assert_array_equal(repeated,out[0]);outputs[device]=np.stack(out);result[device]={'image_seconds':times,'median_image_seconds':float(np.median(times)),'repeat_error':0.,'resource':resources()}
  drift=float(abs(outputs['cpu']-outputs['mps']).max());assert drift<p['tolerance']['CPU_MPS_feature_maxabs'],drift;assert initial==statehash(m);assert not m.training and all(v.grad is None and not v.requires_grad for v in m.parameters());del m;torch.mps.empty_cache()
  with (_repo_path(R, 'benchmark-features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=p['smoke_IDs'],**outputs)
 rec={'passed':True,'rows':3,'batch':1,'dtype':'float32','parameters':params,'devices':result,'CPU_MPS_max_abs':drift,'weights_unchanged':True,'seconds':time.perf_counter()-start,'projected400_MPS_seconds':result['mps']['median_image_seconds']*400,'feature_sha256':sha(_repo_path(R, 'benchmark-features.npz')),'no_labels_read':True,'scope':'3 originalfold0-training images only'};assert rec['projected400_MPS_seconds']<p['resources']['max_feature400_seconds'];write(_repo_path(R, 'benchmark.json'),rec);emit(rec)

def extract():
 p=check();assert read(_repo_path(R, 'benchmark.json'))['passed'];scope_guard(p);start=time.perf_counter();rows=[];hashes={}
 with Monitor('extraction',p['resources']) as mon:
  m=model();initial=statehash(m);m=m.to('mps')
  for j,i in enumerate(p['IDs']):
   rows.append(feature(m,torch.from_numpy(image_input(i)[None]).to('mps')));hashes[i]=sha(_repo_path(P, 'data/train', f'{i}.png'))
   if (j+1)%40==0:mon.sample();emit({'L_features_rows':j+1,'seconds':time.perf_counter()-start,'resource':mon.trace[-1]})
  X=np.stack(rows).astype(float);assert X.shape==(400,9216) and np.isfinite(X).all();assert statehash(m)==initial;assert all(v.grad is None for v in m.parameters());del m;torch.mps.empty_cache()
  with np.load(_repo_path(R, 'benchmark-features.npz'),allow_pickle=False) as z:
   for j,i in enumerate(z['ids']):np.testing.assert_array_equal(X[p['IDs'].index(i)],z['mps'][j])
  old=read(_repo_path(P, 'artifacts/dinob-062/extraction.json'));assert hashes==old['image_sha256']
  with (_repo_path(R, 'features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=p['IDs'],X=X)
 rec={'passed':True,'shape':list(X.shape),'seconds':time.perf_counter()-start,'feature_sha256':sha(_repo_path(R, 'features.npz')),'image_sha256':hashes,'weight_sha256':read(_repo_path(R, 'provenance.json'))['sha256'],'model_state_unchanged':True,'benchmark3_exact':True,'raw_image_hashes_match_B':True,'batch':1,'dtype':'float32','no_labels_or_holdout_test_read':True};write(_repo_path(R, 'extraction.json'),rec);emit({k:v for k,v in rec.items() if k!='image_sha256'})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['benchmark','extract']);mode=ap.parse_args().mode
 try:globals()[mode]()
 except Exception as e:
  write(_repo_path(R, f'{mode}-failure.json'),{'error':repr(e),'type':type(e).__name__});raise
