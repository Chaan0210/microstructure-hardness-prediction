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

plan=read(_repo_path(R, 'plan.json'));cfg=plan['recipe'];SPLITS=['20261031','20261101'];ARMS=['detach','true','permuted']
def check():
 for n,h in read(_repo_path(R, 'preservation.json')).items():assert sha(_repo_path(P, n))==h,n
 for n,h in plan['inputs_SHA256'].items():assert sha(_repo_path(P, n))==h,n
 for n,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, n))==h,n
 assert all(q['num_threads']<=2 for q in threadpool_info())
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute();mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(q in mode for q in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
  if writing and not p.is_relative_to(R):raise RuntimeError('201writes only')
  if p.is_relative_to(_repo_path(P, 'data')) or p==_repo_path(P, 'experiments/candidate022-198/train-input.npz') or p.name=='candidate022-public-submission.json':raise RuntimeError('Rawdata/holdout/test/full500labels/Publicscore excluded201')
sys.addaudithook(audit)
def data(split):
 assert split in SPLITS;d=load(_repo_path(P, f'experiments/dinol-177/{split}/predictions.npz'));x=load(_repo_path(P, 'artifacts/run-012/development_features.npz'));a=load(_repo_path(P, 'artifacts/nested-calibration-074/development_features.npz'));old=load(_repo_path(P, f'artifacts/cnn-seed-mean-133/audit/{split}.npz'))
 for key in ['ids','y']:np.testing.assert_array_equal(d[key],x[key]);np.testing.assert_array_equal(d[key],old[key])
 np.testing.assert_array_equal(d['g'],x['groups']);np.testing.assert_array_equal(d['fold'],old['fold']);np.testing.assert_array_equal(d['ids'],a['ids']);assert d['ids'].tolist()==plan['IDs'] and len(set(d['ids']))==400 and len(set(d['g']))==400;d['images']=a['images'];assert d['images'].shape==(400,1,128,128) and d['images'].dtype==np.float32 and np.isfinite(d['images']).all();d['best021']=d['blend_L_MLP'];np.testing.assert_array_equal(d['best021'],.9*d['reference018']+.1*d['L_MLP']);d['old_CNN']=old['seeds'][0];d['old_CNN_seeds']=old['seeds'];d['split']=split
 for k in range(5):
  T,V=indices(d,k);assert T.tolist()==plan['folds'][split][str(k)]['train'] and V.tolist()==plan['folds'][split][str(k)]['validation'];assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V])
 return d
def indices(d,k):return np.flatnonzero(d['fold']!=k),np.flatnonzero(d['fold']==k)
def preprocessing(d,T,k):
 norm={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())};edges=np.quantile(d['y'][T],[.25,.5,.75],method='linear');assert np.all(np.diff(edges)>0) and norm['y_std']>0
 bins=np.searchsorted(edges,d['y'][T],side='right');permutation=np.random.default_rng(cfg['shuffle_seed']+k).permutation(len(T));shuffled=bins[permutation];assert set(bins)==set(range(4));np.testing.assert_array_equal(np.bincount(bins,minlength=4),np.bincount(shuffled,minlength=4));return norm,edges,bins,permutation,shuffled
def old_checkpoint(split,k):return _repo_path(P, f'artifacts/frozen-validation-021/seed-{split}/fold-{k}/cnn-seed20261003.pt')
@torch.no_grad()
def legacy_predict(obj,images,ii,device):
 m=legacy.TinyCNN().to(device);m.load_state_dict({n:v for n,v in obj['state_dict'].items() if not n.startswith('aux.')});m.eval();nn=obj['normalization'];out=[]
 for j in range(0,len(ii),32):
  x=np.ascontiguousarray((images[ii[j:j+32]]-nn['x_mean'])/max(nn['x_std'],1e-6));out.extend((m(torch.from_numpy(x).to(device)).cpu().numpy()*nn['y_std']+nn['y_mean']).tolist())
 del m;return np.array(out)
def canonical_predictions(d,k,folder):
 obj=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);T,V=indices(d,k);pp=legacy_predict(obj,d['images'],V,'mps');tp=legacy_predict(obj,d['images'],T,'mps');old=load(folder/'predictions.npz');norm=obj['normalization'];recovered=((old['validation_predictions']-norm['y_mean'])/norm['y_std']).astype(np.float32);converted=(recovered*norm['y_std']+norm['y_mean']).astype(float);diff=float(abs(converted-pp).max());assert diff<.001;arrays(folder/'legacy-predictions.npz',ids=d['ids'][V],indices=V,validation_prediction=pp,train_indices=T,train_prediction=tp);write(folder/'precision.json',{'canonical':'batch32 legacyfloat32targetdenormalization;193internaldiagnostic batch64float64 preserved','batch64_converted_vs_batch32_max_error':diff,'finite':bool(np.isfinite(pp).all())});return pp

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
