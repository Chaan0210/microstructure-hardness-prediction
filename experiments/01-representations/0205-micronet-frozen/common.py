from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time,socket,subprocess,re,resource,threading
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
assert Path(sys.prefix).resolve()==_repo_path(P, 'experiments/pretrained-cnn-185/.venv')
network_attempts=[]
def blocked(*a,**k):network_attempts.append('blocked');raise RuntimeError('Network disabled during 205 execution')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np,torch,torchvision
from torch import nn
from PIL import Image
from threadpoolctl import threadpool_limits,threadpool_info
threadpool_limits(2);torch.set_num_threads(2);torch.set_num_interop_threads(1)
assert torch.__version__.split('+')[0]=='2.14.0' and torchvision.__version__.split('+')[0]=='0.29.0'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def emit(v):print(json.dumps(v,ensure_ascii=False),flush=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def arrays(p,**v):
 with Path(p).open('xb') as f:np.savez_compressed(_repo_io(f),**v)
def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def statehash(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
plan=read(_repo_path(R, 'plan.json'))
allowed={_repo_path(P, 'data/train', f'{i}.png') for i in plan['IDs']}
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if p.is_relative_to(_repo_path(P, 'data')) and p not in allowed:raise RuntimeError('Only development allowlist raw images are allowed')
  if p.name in ['.env','.netrc','.git-credentials','credentials','token']:raise RuntimeError('Credential access prohibited')
sys.addaudithook(audit)
def check():
 for n,h in read(_repo_path(R, 'preservation.json')).items():assert sha(_repo_path(P, n))==h,n
 for n,h in plan['input_sha256'].items():assert sha(_repo_path(P, n))==h,n
 if (_repo_path(R, 'code-lock.json')).exists():
  for n,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, n))==h,n
 assert not network_attempts
 assert all(x['num_threads']<=2 for x in threadpool_info())
def model(arm):
 m=torchvision.models.resnet50(weights=None);m.fc=nn.Identity()
 sd=torch.load(_repo_path(P, plan['weights'][arm]['path']),map_location='cpu',weights_only=True)
 assert type(sd) in [dict,__import__('collections').OrderedDict] and all(torch.is_tensor(v) for v in sd.values())
 if arm=='imagenet':
  assert set(sd)-set(m.state_dict())=={'fc.weight','fc.bias'};sd={k:v for k,v in sd.items() if not k.startswith('fc.')}
 else:assert set(sd)==set(m.state_dict())
 assert all(sd[k].shape==v.shape and sd[k].dtype==v.dtype for k,v in m.state_dict().items())
 result=m.load_state_dict(sd,strict=True);assert not result.missing_keys and not result.unexpected_keys
 m.requires_grad_(False);m.eval();return m

def image_input(i):
 assert i in plan['IDs']
 with Image.open(_repo_path(P, 'data/train', f'{i}.png')) as im:
  assert im.mode=='L' and im.size==(256,256);a=np.asarray(im,dtype=np.float32)/255.
 a=(np.repeat(a[None],3,0)-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
 return torch.from_numpy(a[None].copy())
def descriptor(m,x):
 h=m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))));assert h.shape[1:]==(2048,8,8)
 return m.avgpool(h).flatten(1)
def data():
 d=load(_repo_path(P, 'experiments/dinol-177/original/predictions.npz'));z=load(_repo_path(P, 'artifacts/run-012/development_features.npz'))
 for a,b in [('ids','ids'),('y','y'),('groups','g')]:np.testing.assert_array_equal(z[a],d[b])
 assert d['ids'].tolist()==plan['IDs'];np.testing.assert_array_equal(d['blend_L_MLP'],.9*d['reference018']+.1*d['L_MLP'])
 for k,f in plan['folds'].items():
  T=np.array(f['train']);V=np.array(f['validation']);np.testing.assert_array_equal(V,np.flatnonzero(d['fold']==int(k)));assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V])
 return d

def swap():
 s=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',s).groups();return int(float(v)*(1024**2 if u=='M' else 1024**3))
def resources():
 rss=int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024;driver=torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else 0
 return {'RSS':rss,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver':driver,'RSS_plus_MPS':rss+driver,'swap':swap()}
class Monitor:
 def __init__(self,name,seconds):self.name=name;self.seconds=seconds;self.start=time.monotonic();self.samples=[];self.initial=swap();self.stop=threading.Event()
 def sample(self):
  v={'seconds':time.monotonic()-self.start,**resources()};self.samples.append(v);bad=[]
  if max(v['RSS_plus_MPS'],v['peak_RSS'])>6e9:bad.append('6 GB limit')
  if v['swap']>self.initial:bad.append('Swap growth')
  if v['seconds']>self.seconds:bad.append('Time limit')
  if bad:write(_repo_path(R, self.name+'-STOP.json'),{'reasons':bad,'samples':self.samples});os._exit(74)
 def loop(self):
  while not self.stop.wait(1):self.sample()
 def __enter__(self):self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
 def __exit__(self,*a):self.stop.set();self.thread.join();self.sample();write(_repo_path(R, self.name+'-resources.json'),{'samples':self.samples,'initial_swap':self.initial})
