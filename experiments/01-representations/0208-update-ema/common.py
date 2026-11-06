from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time,subprocess,re,resource,threading,socket,gc,csv
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
assert Path(sys.prefix).resolve()==_repo_path(P, '.venv')
os.environ['TMPDIR']=str(_repo_path(R, 'tmp'));os.environ['MPLCONFIGDIR']=str(_repo_path(R, 'cache/matplotlib'))
def blocked(*a,**k):raise RuntimeError('Network forbidden in208')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np,torch
from PIL import Image,ImageDraw
from threadpoolctl import threadpool_limits,threadpool_info
threadpool_limits(2);torch.set_num_threads(2)
sys.path.insert(0,str(_repo_path(P, 'studies')));import cnn_controls as legacy;sys.path.remove(str(_repo_path(P, 'studies')))
os.environ['TORCH_HOME']=str(_repo_path(R, 'cache/torch'))
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def emit(v):print(json.dumps(v,ensure_ascii=False),flush=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(p,keys=None):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def arrays(p,**v):
 with Path(p).open('xb') as f:np.savez_compressed(_repo_io(f),**v)
def statehash(s):
 h=hashlib.sha256()
 for n,v in sorted(s.items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
plan=read(_repo_path(R, 'plan.json'));device='mps' if torch.backends.mps.is_available() else 'cpu'
access={'raw_data_attempts':0,'outside_writes':0}
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if p.is_relative_to(_repo_path(P, 'data')):access['raw_data_attempts']+=1;raise RuntimeError('All rawdata forbidden;208 dev400 caches only')
  if p.name in ['.env','.netrc','.git-credentials','credentials','token']:raise RuntimeError('Credential access forbidden')
  mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(z in mode for z in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
  if writing and not p.is_relative_to(R) and str(p)!='/dev/null':access['outside_writes']+=1;raise RuntimeError('Write outside208 forbidden: '+str(p))
sys.addaudithook(audit)
def check():
 for n,h in read(_repo_path(R, 'preservation.json')).items():assert sha(_repo_path(P, n))==h,n
 for n,h in plan['inputs_SHA256'].items():assert sha(_repo_path(P, n))==h,n
 for file in ['code-lock.json']:
  if (_repo_path(R, file)).exists():
   for n,h in read(_repo_path(R, file)).items():assert sha(_repo_path(R, n))==h,n
 assert all(z['num_threads']<=2 for z in threadpool_info())
 assert access=={'raw_data_attempts':0,'outside_writes':0}
def images():
 z=load(_repo_path(P, plan['image_cache']),['ids','images']);assert z['ids'].tolist()==plan['IDs'];x=z['images'];assert x.shape==(400,1,128,128) and x.dtype==np.float32 and np.isfinite(x).all();return z['ids'],x
def data():
 ids,x=images();d=load(_repo_path(P, 'experiments/dinol-177/original/predictions.npz'));z=load(_repo_path(P, 'artifacts/run-012/development_features.npz'),['ids','y','groups'])
 for key in ['ids','y']:np.testing.assert_array_equal(d[key],z[key])
 np.testing.assert_array_equal(d['g'],z['groups']);np.testing.assert_array_equal(ids,d['ids']);assert len(set(ids))==400 and len(set(d['g']))==400;d['images']=x
 for k,f in plan['folds'].items():
  T,V=indices(int(k));assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V]);np.testing.assert_array_equal(V,np.flatnonzero(d['fold']==int(k)))
 return d
def indices(k):return tuple(np.array(plan['folds'][str(k)][v],dtype=int) for v in ['train','validation'])
def norm(d,T):return {'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())}

def rnghash():
 out={'CPU':ah(torch.get_rng_state().numpy())}
 if device=='mps':out['MPS']=ah(torch.mps.get_rng_state().numpy())
 return out
def sync():
 if device=='mps':torch.mps.synchronize()
def resources():
 rss=int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024
 q=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',q).groups();sw=int(float(v)*(1024**2 if u=='M' else 1024**3));mps=torch.mps.driver_allocated_memory() if device=='mps' else 0
 return {'RSS':rss,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS':mps,'conservative_RSS_plus_MPS':max(rss,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)+mps,'swap':sw}
class Monitor:
 def __init__(self,name,limit):self.name=name;self.limit=limit;self.start=time.monotonic();self.samples=[];self.stop=threading.Event();self.initial=resources()['swap']
 def sample(self):
  v={'seconds':time.monotonic()-self.start,**resources()};self.samples.append(v);bad=[]
  if v['conservative_RSS_plus_MPS']>6e9:bad.append('6GB')
  if v['swap']>self.initial:bad.append('swap growth')
  if v['seconds']>self.limit:bad.append('time limit')
  if bad:write(_repo_path(R, self.name+'-STOP.json'),{'reasons':bad,'trace':self.samples});os._exit(74)
 def loop(self):
  while not self.stop.wait(1):self.sample()
 def __enter__(self):self.sample();self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start();return self
 def __exit__(self,*a):self.stop.set();self.worker.join();self.sample();write(_repo_path(R, self.name+'-resources.json'),{'trace':self.samples,'initial_swap':self.initial,'access':access})
