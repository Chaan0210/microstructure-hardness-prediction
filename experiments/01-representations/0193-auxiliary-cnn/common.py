from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib,time,subprocess,re,resource,threading,socket
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
assert Path(sys.prefix).resolve()==_repo_path(P, '.venv')
def blocked(*a,**k):raise RuntimeError('Network disabled in193')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np,torch
from threadpoolctl import threadpool_limits,threadpool_info
torch.set_num_threads(2);threadpool_limits(2)
sys.path.insert(0,str(_repo_path(P, 'studies')))
import cnn_controls as legacy
os.environ['TORCH_HOME']=str(_repo_path(R, 'cache/torch'))
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,indent=2,allow_nan=False)
def emit(z):print(json.dumps(z),flush=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def arrays(p,**z):
 with Path(p).open('xb') as f:np.savez_compressed(_repo_io(f),**z)
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def statehash(s):
 h=hashlib.sha256()
 for n,v in sorted(s.items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
plan=read(_repo_path(R, 'plan.json'));cfg=plan['recipe']
def check():
 for n,h in read(_repo_path(R, 'preservation.json')).items():assert sha(_repo_path(P, n))==h,n
 for n,h in plan['input_sha256'].items():assert sha(_repo_path(P, n))==h,n
 for n,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, n))==h,n
 assert all(q['num_threads']<=2 for q in threadpool_info())
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if p.is_relative_to(_repo_path(P, 'data')):raise RuntimeError('Raw data forbidden;193 cacheddev400 only')
  if p.name in ['.env','.netrc','.git-credentials','credentials','token'] and not p.is_relative_to(R):raise RuntimeError('Credential access forbidden')
sys.addaudithook(audit)
def data():
 d=load(_repo_path(P, 'experiments/dinol-177/original/predictions.npz'));z=load(_repo_path(P, 'artifacts/run-012/development_features.npz'));a=load(_repo_path(P, plan['image_cache']));m=read(_repo_path(P, 'artifacts/run-001/split_manifest.json'))
 for key in ['ids','y']:np.testing.assert_array_equal(d[key],z[key])
 np.testing.assert_array_equal(d['g'],z['groups']);assert d['ids'].tolist()==a['ids'].tolist()==plan['IDs']==sorted(m['development_folds']);assert not set(d['ids'])&set(m['holdout']);assert len(set(d['ids']))==400
 np.testing.assert_array_equal(d['fold'],[m['development_folds'][i] for i in d['ids']]);np.testing.assert_array_equal(d['blend_L_MLP'],.9*d['reference018']+.1*d['L_MLP']);assert abs(rmse(d['y'],d['blend_L_MLP'])-12.786200775750695)<1e-12
 d['images']=a['images'];assert d['images'].shape==(400,1,128,128) and d['images'].dtype==np.float32 and np.isfinite(d['images']).all()
 for k in range(5):
  T,V=indices(d,k);assert T.tolist()==plan['folds'][str(k)]['train'] and V.tolist()==plan['folds'][str(k)]['validation'];assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V])
 return d
def indices(d,k):return np.flatnonzero(d['fold']!=k),np.flatnonzero(d['fold']==k)
def preprocessing(d,T,k):
 norm={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())};edges=np.quantile(d['y'][T],[.25,.5,.75],method='linear');assert np.all(np.diff(edges)>0) and norm['y_std']>0
 bins=np.searchsorted(edges,d['y'][T],side='right');permutation=np.random.default_rng(cfg['shuffle_seed']+k).permutation(len(T));shuffled=bins[permutation];assert set(bins)==set(range(4));np.testing.assert_array_equal(np.bincount(bins,minlength=4),np.bincount(shuffled,minlength=4))
 return norm,edges,bins,permutation,shuffled
def swap():
 s=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',s).groups();return int(float(v)*(1024**2 if u=='M' else 1024**3))
def resources():
 rss=int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024;return {'RSS':rss,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS':torch.mps.driver_allocated_memory(),'swap':swap()}
class Monitor:
 def __init__(self,name,limit):self.name=name;self.limit=limit;self.start=time.perf_counter();self.samples=[];self.initialswap=swap();self.stop=threading.Event()
 def sample(self):
  q={'seconds':time.perf_counter()-self.start,**resources()};self.samples.append(q);reasons=[]
  if max(q['RSS'],q['peak_RSS'])+q['MPS']>6e9:reasons.append('6GB')
  if q['swap']>self.initialswap:reasons.append('swap_growth')
  if q['seconds']>self.limit:reasons.append('time_budget')
  if reasons:write(_repo_path(R, self.name+'-STOP.json'),{'reasons':reasons,'trace':self.samples});os._exit(74)
 def loop(self):
  while not self.stop.wait(.5):self.sample()
 def __enter__(self):self.sample();self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start();return self
 def __exit__(self,*args):self.stop.set();self.worker.join();self.sample();write(_repo_path(R, self.name+'-resources.json'),{'trace':self.samples,'initial_swap':self.initialswap})
