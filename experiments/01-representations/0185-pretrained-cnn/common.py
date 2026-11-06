from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib,time,subprocess,re,resource,threading,socket
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
assert Path(sys.prefix).resolve()==_repo_path(R, '.venv')
network_attempts=[]
def blocked(*a,**kw):network_attempts.append('blocked');raise RuntimeError('Networkdisabledafterapproveddownload')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np,torch,torchvision
from torch import nn
from PIL import Image
from threadpoolctl import threadpool_limits
threadpool_limits(2);torch.set_num_threads(2);torch.set_num_interop_threads(1)
assert torch.__version__.split('+')[0]=='2.14.0' and torchvision.__version__.split('+')[0]=='0.29.0'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
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
def statehash(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def check():
 p=read(_repo_path(R, 'plan.json'))
 for n,h in read(_repo_path(R, 'preservation.json')).items():assert sha(_repo_path(P, n))==h,n
 for n,h in p['input_sha256'].items():assert sha(_repo_path(P, n))==h,n
 for n,z in read(_repo_path(R, 'provenance.json'))['weights'].items():assert sha(_repo_path(P, z['path']))==z['sha256']
 for n,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, n))==h,n
 return p
plan=read(_repo_path(R, 'plan.json'));allowed={_repo_path(P, 'data/train', f'{i}.png') for i in plan['IDs']}
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if p.is_relative_to(_repo_path(P, 'data')) and p not in allowed:raise RuntimeError('Holdout/test/rawlabel access forbidden '+str(p))
  if p.name in ['.env','.netrc','.git-credentials','credentials','token'] and not p.is_relative_to(R):raise RuntimeError('Credential access forbidden')
sys.addaudithook(audit)
def image_input(i):
 assert i in plan['IDs']
 with Image.open(_repo_path(P, 'data/train', f'{i}.png')) as im:
  assert im.mode=='L' and im.size==(256,256)
  a=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),dtype=np.float32)/255.
 a=(np.repeat(a[None],3,0)-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
 return torch.from_numpy(a[None].copy())
def model(name):
 if name=='resnet50':
  m=torchvision.models.resnet50(weights=None);filename='resnet50-11ad3fa6.pth'
 else:
  assert name=='efficientnet_b0';m=torchvision.models.efficientnet_b0(weights=None);filename='efficientnet_b0_rwightman-7f5810bc.pth'
 m.load_state_dict(torch.load(_repo_path(R, 'weights', filename),map_location='cpu',weights_only=True),strict=True);m.requires_grad_(False);m.eval()
 if name=='resnet50':prefix=nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool,m.layer1,m.layer2,m.layer3);tail=nn.Sequential(m.layer4,m.avgpool,nn.Flatten(1))
 else:prefix=m.features[:7];tail=nn.Sequential(m.features[7:],m.avgpool,nn.Flatten(1))
 return prefix.eval(),tail.eval()
def swap():
 s=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',s).groups();return int(float(v)*(1024**2 if u=='M' else 1024**3))
def resources():
 rss=int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024;driver=torch.mps.driver_allocated_memory();return {'RSS':rss,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver':driver,'RSS_plus_MPS':rss+driver,'swap':swap()}
class Monitor:
 def __init__(self,name,limit):self.name=name;self.limit=limit;self.start=time.monotonic();self.samples=[];self.initialswap=swap();self.stop=threading.Event()
 def sample(self):
  z={'seconds':time.monotonic()-self.start,**resources()};self.samples.append(z);fail=[]
  if max(z['RSS_plus_MPS'],z['peak_RSS'])>6000000000:fail.append('6GB')
  if z['swap']>self.initialswap:fail.append('swap_growth')
  if z['seconds']>self.limit:fail.append('time_budget')
  if fail:write(_repo_path(R, f'{self.name}-STOP.json'),{'fail':fail,'trace':self.samples});emit({'STOP':fail});os._exit(74)
 def loop(self):
  while not self.stop.wait(2):
   try:self.sample()
   except BaseException as e:emit({'monitor_error':repr(e)});os._exit(75)
 def __enter__(self):self.sample();self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start();return self
 def __exit__(self,*args):self.stop.set();self.worker.join();self.sample();write(_repo_path(R, f'{self.name}-resources.json'),{'trace':self.samples,'initial_swap':self.initialswap})
def data(split):
 c=load(_repo_path(P, 'artifacts/run-012/development_features.npz'));z=load(_repo_path(P, f'experiments/dinol-177/{split}/predictions.npz'))
 for a,b in [('ids','ids'),('y','y'),('groups','g')]:np.testing.assert_array_equal(c[a],z[b])
 assert z['ids'].tolist()==plan['IDs'] and len(z['y'])==400
 np.testing.assert_array_equal(z['blend_L_MLP'],.9*z['reference018']+.1*z['L_MLP'])
 for k in range(5):assert not set(z['g'][z['fold']==k])&set(z['g'][z['fold']!=k])
 return z
