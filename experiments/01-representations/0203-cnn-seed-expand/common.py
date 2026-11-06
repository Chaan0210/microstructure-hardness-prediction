from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib,time,resource,subprocess,re,threading
from pathlib import Path
O=Path(__file__).resolve().parent;R=_RepoPath(__file__).resolve().parents[3]
assert Path.cwd()==R and Path(sys.prefix).resolve()==_repo_path(R, '.venv')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(O, 'cache'));os.environ['TORCH_HOME']=str(_repo_path(O, 'cache/torch'))
import numpy as np,torch
from threadpoolctl import threadpool_limits,threadpool_info
torch.set_num_threads(2);threadpool_limits(2)
sys.path.insert(0,str(_repo_path(R, 'studies')))
import frozen_validation as fv
os.environ['TORCH_HOME']=str(_repo_path(O, 'cache/torch'))
PLAN=json.loads((_repo_path(O, 'plan.json')).read_text());SPLITS=PLAN['splits'];READS=set();BLOCKED=[]
def H(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def J(p):return json.loads(Path(p).read_text())
def W(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def Z(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def arrays(p,**kw):
 with Path(p).open('xb') as f:np.savez_compressed(_repo_io(f),**kw)
def emit(x):sys.__stdout__.write(json.dumps(x,ensure_ascii=False)+'\n');sys.__stdout__.flush()
def ah(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def sh(s):
 h=hashlib.sha256()
 for n,v in sorted(s.items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def cpu(v):
 if torch.is_tensor(v):return v.detach().cpu().clone()
 if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
 if isinstance(v,list):return [cpu(x) for x in v]
 if isinstance(v,tuple):return tuple(cpu(x) for x in v)
 return v
def audit(ev,args):
 if ev.startswith('socket.'):raise RuntimeError('Network forbidden203')
 if ev=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute();mode=args[1];flags=args[2]
  wr=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
  if wr and not p.is_relative_to(O):BLOCKED.append(str(p));raise RuntimeError('Writes only203')
  if p.is_relative_to(R):
   rel=str(p.relative_to(R));runtime=p.is_relative_to(_repo_path(R, '.venv')) or (p.parent==_repo_path(R, 'studies') and p.suffix=='.py') or p.is_relative_to(_repo_path(R, 'studies/src'))
   if not p.is_relative_to(O) and rel not in PLAN['inputs_sha256'] and rel not in PLAN['protected_sha256'] and not(runtime and not wr):BLOCKED.append(rel);raise RuntimeError('Unplanned project access:'+rel)
   READS.add(rel)
  elif str(p).startswith('/Users/'):BLOCKED.append('other_user_path');raise RuntimeError('Outside project203')
sys.addaudithook(audit)
def check():
 assert torch.backends.mps.is_available()
 for p,h in {**PLAN['inputs_sha256'],**PLAN['protected_sha256']}.items():assert H(_repo_path(R, p))==h,p
 lock=J(_repo_path(O, 'code-lock.json'));assert lock['plan_sha256']==H(_repo_path(O, 'plan.json'))
 for p,h in lock['sources'].items():assert H(_repo_path(O, p))==h,p
 assert all(x['num_threads']<=2 for x in threadpool_info())
def data():
 z=Z(_repo_path(R, 'experiments/simplification-202/predictions.npz'))
 with np.load(_repo_path(R, 'artifacts/nested-calibration-074/development_features.npz'),allow_pickle=False) as a:
  np.testing.assert_array_equal(a['ids'],z['ids']);images=a['images'].copy()
 assert images.shape==(400,1,128,128) and images.dtype==np.float32 and np.isfinite(images).all()
 assert z['R'].shape==(3,400) and z['CNN_seeds'].shape==(3,3,400)
 assert ah(z['ids'])==PLAN['unique_ID_order_sha256'] and len(set(z['groups']))==400
 return z,images
def indices(z,si,k):
 T=np.flatnonzero(z['fold'][si]!=k);V=np.flatnonzero(z['fold'][si]==k)
 assert T.tolist()==PLAN['folds'][SPLITS[si]][str(k)]['train'] and V.tolist()==PLAN['folds'][SPLITS[si]][str(k)]['validation'];assert len(T)==320 and len(V)==80 and not set(z['groups'][T])&set(z['groups'][V]);return T,V
def normalization(images,y,T):return {'x_mean':float(images[T].mean()),'x_std':float(images[T].std()),'y_mean':float(y[T].mean()),'y_std':float(y[T].std())}
@torch.no_grad()
def predict(obj,images,ix,device):
 m=fv.engine.TinyCNN().to(device);m.load_state_dict(obj['state_dict']);m.eval();nn=obj['normalization'];out=[]
 for start in range(0,len(ix),32):
  xb=torch.from_numpy((images[ix[start:start+32]]-nn['x_mean'])/max(nn['x_std'],1e-6)).to(device)
  out.extend((m(xb).cpu().numpy()*nn['y_std']+nn['y_mean']).tolist())
 del m;return np.asarray(out)
def swap():
 t=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);value,unit=re.search(r'used = ([0-9.]+)([MG])',t).groups();return int(float(value)*(1024**2 if unit=='M' else 1024**3))
def resource_sample():return {'RSS':int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True))*1024,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver':int(torch.mps.driver_allocated_memory()),'swap':swap()}
class Monitor:
 def __init__(self,name,limit):self.name=name;self.limit=limit;self.start=time.perf_counter();self.samples=[];self.stop=threading.Event();self.initialswap=swap()
 def sample(self):
  x={'seconds':time.perf_counter()-self.start,**resource_sample()};self.samples.append(x);reasons=[]
  if max(x['RSS'],x['peak_RSS'])+x['MPS_driver']>6e9:reasons.append('6GB')
  if x['swap']>self.initialswap:reasons.append('swap_growth')
  if x['seconds']>self.limit:reasons.append('time_budget')
  if reasons:W(_repo_path(O, self.name+'-STOP.json'),{'reasons':reasons,'samples':self.samples});os._exit(74)
 def loop(self):
  while not self.stop.wait(1):self.sample()
 def __enter__(self):self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
 def __exit__(self,*args):self.stop.set();self.thread.join();self.sample();W(_repo_path(O, self.name+'-resources.json'),{'samples':self.samples,'initial_swap':self.initialswap})
def access_report(name):W(_repo_path(O, name),{'project_paths_opened':sorted(READS),'blocked':BLOCKED,'network':False,'scope':'Only predeclared inputs/protectedhashes,existing code/runtime,and own203 files; data/test/holdout paths not allowed.'})
