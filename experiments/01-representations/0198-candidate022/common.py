from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,hashlib,time,subprocess,re,resource,threading,socket,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];OLD=_repo_path(P, 'experiments/reproduction-018-176');PHASE=Path(sys.argv[0]).stem
assert Path(sys.prefix).resolve()==_repo_path(P, '.venv')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
def blocked(*a,**k):raise RuntimeError('Network or fitting forbidden during this phase')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np,torch
from PIL import Image
from threadpoolctl import threadpool_limits,threadpool_info
torch.set_num_threads(2);threadpool_limits(2);sys.path.insert(0,str(_repo_path(P, 'studies')));import cnn_controls as legacy
os.environ['TORCH_HOME']=str(_repo_path(R, 'cache/torch'))
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False,allow_nan=False)
def emit(z):print(json.dumps(z,ensure_ascii=False),flush=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def statehash(s):
 h=hashlib.sha256()
 for n,v in sorted(s.items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def load(p,keys=None):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def arrays(p,**z):
 with Path(p).open('xb') as f:np.savez_compressed(_repo_io(f),**z)
def equal(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
plan=read(_repo_path(R, 'plan.json'));cfg=plan['recipe']
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute();mode=args[1];flags=args[2]
  writing=(isinstance(mode,str) and any(q in mode for q in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
  allowed=[_repo_path(P, 'artifacts/submission-candidate-022.csv'),_repo_path(P, 'artifacts/submission-candidate-022.manifest.json'),_repo_path(P, 'reports/candidate022-198-summary.json')]
  if writing and not (p.is_relative_to(R) or (PHASE=='finalize' and p in allowed)):raise RuntimeError('Write outside198 blocked: '+str(p))
  if p.is_relative_to(_repo_path(P, 'data/test')) or p==_repo_path(P, 'data/sample_submission.csv') or p==_repo_path(OLD, 'test-run/test-features.npz'):
   if PHASE not in ['infer','finalize']:raise RuntimeError('Test read before final training validation blocked')
  if p.name in ['.env','.netrc','.git-credentials','credentials','token'] and not p.is_relative_to(R):raise RuntimeError('Credential access forbidden')
sys.addaudithook(audit)
def check():
 for n,h in read(_repo_path(R, 'preservation.json')).items():
  if PHASE not in ['infer','finalize'] and (n=='experiments/reproduction-018-176/test-run/test-features.npz' or n=='data/sample_submission.csv' or n.startswith('data/test/')):continue
  assert sha(_repo_path(P, n))==h,n
 for n,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, n))==h,n
 assert all(q['num_threads']<=2 for q in threadpool_info())
def normalization(d):
 return {'x_mean':float(d['images'].mean()),'x_std':float(d['images'].std()),'y_mean':float(d['y'].mean()),'y_std':float(d['y'].std())}
def raw_images(ids,split,meta):
 out=[]
 for name in ids:
  p=_repo_path(P, 'data', split, str(name)+'.png');assert sha(p)==meta['image_sha256'][str(name)]
  with Image.open(_repo_io(p)) as im:
   assert im.mode=='L' and im.size==(256,256);out.append(np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)[None]/255.)
 return np.stack(out)
def training_data():
 p=_repo_path(OLD, 'run/train-features.npz');meta=read(_repo_path(OLD, 'run/train-feature-validation.json'));assert meta['passed'] and meta['rows']==500 and sha(p)==meta['feature_sha256'];d=load(p,['ids','images','y']);ids=[f'TRAIN_{j:06d}' for j in range(1,501)];assert d['ids'].tolist()==ids and d['images'].shape==(500,1,128,128) and d['images'].dtype==np.float32 and d['y'].shape==(500,) and np.isfinite(d['y']).all()
 with (_repo_path(P, 'data/train.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
 assert [q['ID'] for q in rows]==ids;np.testing.assert_array_equal(d['y'],[float(q['hardness']) for q in rows]);fresh=raw_images(ids,'train',meta);np.testing.assert_array_equal(fresh,d['images']);assert np.isfinite(fresh).all();return d
@torch.no_grad()
def predict(obj,images,device):
 m=legacy.TinyCNN().to(device);m.load_state_dict({k:v for k,v in obj['state_dict'].items() if not k.startswith('aux.')});m.eval();norm=obj['normalization'];out=[]
 for st in range(0,len(images),32):
  x=np.ascontiguousarray((images[st:st+32]-norm['x_mean'])/max(norm['x_std'],1e-6));z=m(torch.from_numpy(x).to(device)).cpu().numpy();out.extend((z*norm['y_std']+norm['y_mean']).tolist())
 del m
 if device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
 p=np.array(out);assert np.isfinite(p).all();return p

def swap():
 s=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',s).groups();return int(float(v)*(1024**2 if u=='M' else 1024**3))
def resources():return {'RSS':int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver':torch.mps.driver_allocated_memory(),'swap':swap()}
class Monitor:
 def __init__(self,name,limit=600):self.name=name;self.limit=limit;self.start=time.perf_counter();self.samples=[];self.initialswap=swap();self.stop=threading.Event()
 def sample(self):
  q={'seconds':time.perf_counter()-self.start,**resources()};self.samples.append(q);reasons=[]
  if max(q['RSS'],q['peak_RSS'])+q['MPS_driver']>6e9:reasons.append('6GB')
  if q['swap']>self.initialswap:reasons.append('swap_growth')
  if q['seconds']>self.limit:reasons.append('time_budget')
  if reasons:write(_repo_path(R, self.name+'-STOP.json'),{'reasons':reasons,'trace':self.samples});os._exit(74)
 def loop(self):
  while not self.stop.wait(.5):self.sample()
 def __enter__(self):self.sample();self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start();return self
 def __exit__(self,*args):self.stop.set();self.worker.join();self.sample();write(_repo_path(R, self.name+'-resources.json'),{'trace':self.samples,'initial_swap':self.initialswap})
