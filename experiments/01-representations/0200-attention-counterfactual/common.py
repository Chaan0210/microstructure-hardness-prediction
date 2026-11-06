from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib,time,subprocess,re,resource,threading,socket
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];assert Path(sys.prefix).resolve()==_repo_path(P, '.venv')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
def blocked(*a,**k):raise RuntimeError('Training/backward/network forbidden in200')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np,torch
from threadpoolctl import threadpool_limits
threadpool_limits(2);torch.set_num_threads(2);torch.Tensor.backward=blocked;torch.autograd.grad=blocked;torch.optim.AdamW.step=blocked;torch.optim.SGD.step=blocked

def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False,allow_nan=False)
def load(p,keys=None):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def arrays(p,**a):
 with Path(p).open('xb') as f:np.savez_compressed(_repo_io(f),**a)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def statehash(s):
 h=hashlib.sha256()
 for n,v in sorted(s.items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def emit(z):print(json.dumps(z,ensure_ascii=False),flush=True)
plan=read(_repo_path(R, 'plan.json'))
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute();mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(q in mode for q in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
  if writing and not p.is_relative_to(R):raise RuntimeError('New200 files only')
  if p.is_relative_to(_repo_path(P, 'data')) or 'test' in p.name.lower() or p==_repo_path(P, 'experiments/candidate022-198/train-input.npz'):raise RuntimeError('Rawdata/holdout/test/full500labels forbidden200')
sys.addaudithook(audit)
def check():
 for n,h in read(_repo_path(R, 'preservation.json')).items():assert sha(_repo_path(P, n))==h,n
 for n,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, n))==h,n
class TokenReader:
 def __enter__(self):
  p=_repo_path(P, 'artifacts/patch-attention-092/patch_tokens.npy');a=np.load(_repo_io(p),mmap_mode='r',allow_pickle=False);assert a.shape==(400,1296,768) and a.dtype==np.float32;self.offset=a.offset;a._mmap.close();self.file=p.open('rb');return self
 def read(self,ii):
  assert 1<=len(ii)<=2;out=np.empty((len(ii),1296,768),np.float32)
  for n,i in enumerate(ii):self.file.seek(self.offset+int(i)*1296*768*4);raw=self.file.read(1296*768*4);assert len(raw)==1296*768*4;out[n]=np.frombuffer(raw,dtype=np.float32).reshape(1296,768)
  assert np.isfinite(out).all();return out
 def __exit__(self,*a):self.file.close()
def swap():
 s=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',s).groups();return int(float(v)*(1024**2 if u=='M' else 1024**3))
def resources():return {'RSS':int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS':torch.mps.driver_allocated_memory(),'swap':swap()}
class Monitor:
 def __init__(self,name,limit=600):self.name=name;self.limit=limit;self.start=time.perf_counter();self.samples=[];self.initialswap=swap();self.stop=threading.Event()
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
