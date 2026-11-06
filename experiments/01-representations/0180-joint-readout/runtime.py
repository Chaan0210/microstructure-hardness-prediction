from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
os.environ['XFORMERS_DISABLED']='1'
import sys,time,json,hashlib,resource,socket,subprocess,re,threading
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];V=_repo_path(P, 'vendor/dinov2-official/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8');sys.path.insert(0,str(V))
def blocked(*a,**kw):raise RuntimeError('Network/fitting forbidden')
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

def image_input(i,split='train'):
 with Image.open(_repo_path(P, 'data', split, f'{i}.png')) as im:
  assert im.mode=='L' and im.size==(256,256);a=np.asarray(im.resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/255.
 return np.ascontiguousarray((np.repeat(a[None],3,axis=0)-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None])

def model():
 m=dinov2_vitl14(pretrained=False);sd=torch.load(_repo_path(P, 'experiments/dinol-177/dinov2_vitl14_pretrain.pth'),map_location='cpu',weights_only=True);m.load_state_dict(sd,strict=True);del sd;m.requires_grad_(False);m.eval();assert len(m.blocks)==24 and m.embed_dim==1024 and m.num_register_tokens==0;return m

def statehash(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def feature(m,x):
 with torch.inference_mode():
  layers=m.get_intermediate_layers(x,n=[11,17,23],return_class_token=True,norm=True);assert all(p.shape==(1,1296,1024) and c.shape==(1,1024) for p,c in layers);z=torch.cat([torch.cat([c,p.mean(1),p.std(1,correction=0)],1) for p,c in layers],1).cpu().numpy()[0]
 assert z.shape==(9216,) and np.isfinite(z).all();return z

def check():
 p=read(_repo_path(R, "plan.json"))
 for name,h in read(_repo_path(R, "code-lock.json"))["sources"].items():assert sha(_repo_path(R, name))==h,name
 for name,h in {**p["inputs"],**p["protected"]}.items():assert sha(_repo_path(P, name))==h,name
 return p

def inference_guards():
 from sklearn.preprocessing import StandardScaler,MinMaxScaler
 StandardScaler.fit=blocked;StandardScaler.partial_fit=blocked;MinMaxScaler.fit=blocked;MinMaxScaler.partial_fit=blocked;torch.Tensor.backward=blocked
