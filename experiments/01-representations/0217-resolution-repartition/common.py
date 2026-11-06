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
def blocked(*a,**k):raise RuntimeError('Network forbidden in217')
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
  if p.is_relative_to(_repo_path(P, 'data')):access['raw_data_attempts']+=1;raise RuntimeError('All rawdata forbidden;217 dev400 caches only')
  if p.name in ['.env','.netrc','.git-credentials','credentials','token']:raise RuntimeError('Credential access forbidden')
  mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(z in mode for z in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
  if writing and not p.is_relative_to(R) and str(p)!='/dev/null':access['outside_writes']+=1;raise RuntimeError('Write outside217 forbidden: '+str(p))
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
SPLITS=plan['splits']
def data(split,arm='control'):
 assert split in SPLITS;ids,x=images();d=load(_repo_path(P, f'experiments/dinol-177/{split}/predictions.npz'));z=load(_repo_path(P, 'artifacts/run-012/development_features.npz'),['ids','y','groups']);old=load(_repo_path(P, f'artifacts/cnn-seed-mean-133/audit/{split}.npz'))
 for key in ['ids','y']:np.testing.assert_array_equal(d[key],z[key]);np.testing.assert_array_equal(d[key],old[key])
 np.testing.assert_array_equal(d['g'],z['groups']);np.testing.assert_array_equal(ids,d['ids']);np.testing.assert_array_equal(d['fold'],old['fold']);assert len(set(ids))==400 and len(set(d['g']))==400;d['images']=x if arm=='control' else load(_repo_path(P, 'experiments/resolution-216/native256.npz'),['images'])['images'];d['split']=split;d['best021']=d['blend_L_MLP'];d['old_CNN']=old['seeds'][0];d['old_CNN_seeds']=old['seeds']
 np.testing.assert_array_equal(d['best021'],.9*d['reference018']+.1*d['L_MLP'])
 for k in range(5):
  T,V=indices(d,k);assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V]);assert T.tolist()==plan['folds'][split][str(k)]['train'] and V.tolist()==plan['folds'][split][str(k)]['validation']
 return d
def indices(d,k):return np.flatnonzero(d['fold']!=k),np.flatnonzero(d['fold']==k)
def old_checkpoint(split,k):return _repo_path(P, f'artifacts/frozen-validation-021/seed-{split}/fold-{k}/cnn-seed20261003.pt')
def base_state(sd):return {n:v for n,v in sd.items() if not n.startswith('aux.')}
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

def lineage(d):
 split=d['split'];old=load(_repo_path(P, f'artifacts/component-ablation-121/{split}/predictions.npz'));agg=load(_repo_path(P, f'artifacts/bootstrap-selector-144/{split}/predictions.npz'));pipe=load(_repo_path(P, f'artifacts/cnn-seed-mean-133/{split}/predictions.npz'))
 for a in [old,agg,pipe]:
  for key in ['ids','y','fold']:np.testing.assert_array_equal(a[key],d[key])
 comp=old['components'].copy();comp[:,2]=d['old_CNN_seeds'].mean(0);w=np.array([0.,.1875,.1875,.125,.125]);raw=comp@(w/w.sum());np.testing.assert_array_equal(raw,pipe['raw']);base=np.empty(400);coef=np.empty(400);cor=np.empty(400)
 for k in range(5):
  T,V=indices(d,k);cal=plan['calibrators'][split][str(k)];assert cal==read(_repo_path(P, f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/report.json'))['outer_calibrator']==read(_repo_path(P, f'artifacts/cnn-seed-mean-133/{split}/pipeline-fold-{k}/report.json'))['outer_calibrator'];coef[V]=.081*cal['slope'];base[V]=.9*(cal['target_center']+cal['slope']*(raw[V]-cal['prediction_center']))+.1*old['short'][V];q=load(_repo_path(P, f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/predictions.npz'));np.testing.assert_array_equal(q['outer_indices'],V);np.testing.assert_array_equal(base[V],q['base']);parts=[]
  for j in range(3):
   q=load(_repo_path(P, f'artifacts/bootstrap-selector-144/{split}/fold-{k}-block-{j}/predictions.npz'));np.testing.assert_array_equal(q['outer_indices'],V);np.testing.assert_array_equal(q['weights']@q['full_candidate_predictions'],q['primary']);parts.append(q['primary'])
  cor[V]=np.mean(parts,axis=0)
 np.testing.assert_array_equal(base+cor,agg['primary']);np.testing.assert_array_equal(agg['primary'],d['reference018']);np.testing.assert_array_equal(.9*(base+cor)+.1*d['L_MLP'],d['best021']);return {'coefficient':coef,'components':comp,'short':old['short'],'correction':cor,'raw':raw,'base':base,'L_MLP':d['L_MLP']}
