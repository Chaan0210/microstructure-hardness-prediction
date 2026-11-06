from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib,time,subprocess,re,resource,threading,socket
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];assert Path(sys.prefix).resolve()==_repo_path(P, '.venv')
network_attempts=[]
def blocked(*a,**k):network_attempts.append(True);raise RuntimeError('Network forbidden in cached195')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
threadpool_limits(2)
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False,allow_nan=False)
def arrays(p,**a):
 with Path(p).open('xb') as f:np.savez_compressed(_repo_io(f),**a)
def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def emit(q):print(json.dumps(q,ensure_ascii=False),flush=True)
plan=read(_repo_path(R, 'plan.json'));ARMS=plan['arms']
def check():
 for relative,h in read(_repo_path(R, 'preservation.json')).items():assert sha(_repo_path(P, relative))==h,relative
 for relative,h in plan['input_SHA256'].items():assert sha(_repo_path(P, relative))==h,relative
 for relative,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, relative))==h,relative
 assert all(t['num_threads']<=2 for t in threadpool_info())
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  q=Path(os.fsdecode(args[0])).absolute()
  if q.is_relative_to(_repo_path(P, 'data')):raise RuntimeError('Raw data forbidden195')
  if q.name in ['.env','.netrc','.git-credentials','credentials','token'] and not q.is_relative_to(R):raise RuntimeError('Credentials forbidden')
sys.addaudithook(audit)
def data():
 d=load(_repo_path(P, 'experiments/matched-replacement-194/replacement-OOF.npz'));a=load(_repo_path(P, 'experiments/dinol-177/original/predictions.npz'));b=load(_repo_path(P, 'artifacts/run-012/development_features.npz'))
 for k in ['ids','y','fold']:np.testing.assert_array_equal(d[k],a[k])
 for k in ['ids','y','groups']:np.testing.assert_array_equal(d[k],b[k])
 np.testing.assert_array_equal(d['groups'],a['g']);np.testing.assert_array_equal(d['best021'],a['blend_L_MLP']);np.testing.assert_array_equal(a['blend_L_MLP'],.9*a['reference018']+.1*a['L_MLP'])
 assert d['ids'].tolist()==plan['IDs'] and len(set(d['ids']))==400 and len(set(d['groups']))==400
 for k in range(5):
  f=plan['folds'][str(k)];T=np.array(f['train']);V=np.array(f['validation']);np.testing.assert_array_equal(V,np.flatnonzero(d['fold']==k));assert not set(d['groups'][T])&set(d['groups'][V]);assert len(T)==320 and len(V)==80
 return d
def tokens():
 path=_repo_path(P, 'artifacts/patch-attention-092/patch_tokens.npy');z=np.load(_repo_io(path),mmap_mode='r',allow_pickle=False);assert z.shape==(400,1296,768) and z.dtype==np.float32;offset=z.offset;z._mmap.close();del z
 with path.open('rb') as f:
  f.seek(offset)
  for i in range(400):
   raw=f.read(1296*768*4);assert len(raw)==1296*768*4;a=np.frombuffer(raw,dtype=np.float32).reshape(1296,768);assert np.isfinite(a).all();yield i,a
  assert f.read(1)==b''
def features(arm,f=None):
 if f is None:f=load(_repo_path(R, 'features.npz'))
 X=f['baseline'].astype(float)
 if arm!='baseline':X=np.concatenate((X,f['maximum' if arm=='plus_max' else 'minimum'].astype(float)),axis=1)
 assert X.shape==(400,6912 if arm=='baseline' else 7680) and np.isfinite(X).all();return X

def swap():
 q=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',q).groups();return int(float(v)*(1024**2 if u=='M' else 1024**3))
class Monitor:
 def __init__(self,name):self.name=name;self.start=time.perf_counter();self.stop=threading.Event();self.trace=[];self.initialswap=swap()
 def sample(self):
  q={'seconds':time.perf_counter()-self.start,'RSS':int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'swap':swap()};self.trace.append(q)
  if max(q['RSS'],q['peak_RSS'])>6e9 or q['swap']>self.initialswap or q['seconds']>600:write(_repo_path(R, self.name+'-STOP.json'),self.trace);os._exit(74)
 def loop(self):
  while not self.stop.wait(.5):self.sample()
 def __enter__(self):self.sample();self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start();return self
 def __exit__(self,*a):self.stop.set();self.worker.join();self.sample();write(_repo_path(R, self.name+'-resources.json'),{'trace':self.trace,'initial_swap':self.initialswap,'MPS_used':False})
