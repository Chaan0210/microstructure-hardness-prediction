from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
os.environ['XFORMERS_DISABLED']='1'
import sys,time,json,hashlib,resource,socket,subprocess,re,threading
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
def blocked(*a,**kw):raise RuntimeError('Network/fitting forbidden')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np,torch
from PIL import Image
from threadpoolctl import threadpool_limits
threadpool_limits(2);torch.set_num_threads(2)
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

def check():
 p=read(_repo_path(R, "plan.json"))
 for name,h in read(_repo_path(R, "code-lock.json"))["sources"].items():assert sha(_repo_path(R, name))==h,name
 for name,h in {**p["inputs"],**p["protected"]}.items():assert sha(_repo_path(P, name))==h,name
 return p

