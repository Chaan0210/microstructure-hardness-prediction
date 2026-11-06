from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib,time,subprocess,re,resource,threading,socket
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];assert Path(sys.prefix).resolve()==_repo_path(P, '.venv')
def forbidden(*a,**k):raise RuntimeError('Network/training forbidden in cache-only194')
socket.create_connection=forbidden;socket.socket.connect=forbidden;socket.socket.connect_ex=forbidden
import numpy as np,torch
from threadpoolctl import threadpool_limits,threadpool_info
torch.set_num_threads(2);threadpool_limits(2);torch.Tensor.backward=forbidden
plan=json.loads((_repo_path(R, 'plan.json')).read_text());ARMS=plan['arms']
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,indent=2,allow_nan=False,ensure_ascii=False)
def emit(z):print(json.dumps(z,ensure_ascii=False),flush=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def arrays(p,**a):
 with Path(p).open('xb') as f:np.savez_compressed(_repo_io(f),**a)
def check():
 for n,h in read(_repo_path(R, 'preservation.json')).items():assert sha(_repo_path(P, n))==h,n
 for n,h in plan['inputs_SHA256'].items():assert sha(_repo_path(P, n))==h,n
 for n,h in read(_repo_path(R, 'code-lock.json')).items():assert sha(_repo_path(R, n))==h,n
 assert all(z['num_threads']<=2 for z in threadpool_info())
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if p.is_relative_to(_repo_path(P, 'data')):raise RuntimeError('Rawdata/holdout/test forbidden in194')
  if p.name in ['.env','.netrc','.git-credentials','credentials','token'] and not p.is_relative_to(R):raise RuntimeError('Credential access forbidden')
sys.addaudithook(audit)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def data():
 a=load(_repo_path(P, 'experiments/auxiliary-cnn-193/OOF.npz'));b=load(_repo_path(P, 'experiments/dinol-177/original/predictions.npz'));cnn=load(_repo_path(P, 'artifacts/cnn-seed-mean-133/audit/original.npz'));old=load(_repo_path(P, 'artifacts/component-ablation-121/original/predictions.npz'));agg=load(_repo_path(P, 'artifacts/bootstrap-selector-144/original/predictions.npz'));pipe=load(_repo_path(P, 'artifacts/cnn-seed-mean-133/original/predictions.npz'));assert read(_repo_path(P, 'experiments/auxiliary-cnn-193/validation.json'))['passed']
 assert a['ids'].tolist()==plan['IDs'] and len(set(a['ids']))==400 and len(set(a['groups']))==400
 for z in [b,cnn,old,agg,pipe]:
  for k in ['ids','y','fold']:np.testing.assert_array_equal(a[k],z[k])
 np.testing.assert_array_equal(a['groups'],b['g']);np.testing.assert_array_equal(a['best021'],b['blend_L_MLP']);np.testing.assert_array_equal(b['reference018'],agg['primary']);np.testing.assert_array_equal(a['best021'],.9*agg['primary']+.1*b['L_MLP']);np.testing.assert_array_equal(cnn['mean3'],cnn['seeds'].mean(0))
 for folder,name,key in [('experiments/auxiliary-cnn-193','OOF.npz','OOF_SHA256'),('experiments/dinol-177/original','predictions.npz','predictions_sha256'),('artifacts/bootstrap-selector-144/original','predictions.npz','predictions_sha256'),('artifacts/component-ablation-121/original','predictions.npz','predictions_sha256'),('artifacts/cnn-seed-mean-133/original','predictions.npz','predictions_sha256')]:assert sha(_repo_path(P, folder, name))==read(_repo_path(P, folder, 'report.json'))[key]
 for k in range(5):
  T=np.flatnonzero(a['fold']!=k);V=np.flatnonzero(a['fold']==k);assert T.tolist()==plan['folds'][str(k)]['train'] and V.tolist()==plan['folds'][str(k)]['validation'];assert len(T)==320 and len(V)==80 and not set(a['groups'][T])&set(a['groups'][V])
 return {'a':a,'b':b,'cnn':cnn,'old':old,'agg':agg,'pipe':pipe}
def legacy_precision(d):
 a=d['a'];converted={arm:np.empty(400) for arm in ARMS};checks=[]
 for k in range(5):
  V=a['fold']==k;p=torch.load(_repo_path(P, f'experiments/auxiliary-cnn-193/original/fold-{k}-detach/model.pt'),map_location='cpu',weights_only=True);norm=p['normalization'];T=~V;assert p['train_ids']==a['ids'][T].tolist() and p['validation_ids']==a['ids'][V].tolist();assert norm['y_mean']==float(a['y'][T].mean()) and norm['y_std']==float(a['y'][T].std());cert=read(_repo_path(P, f'experiments/auxiliary-cnn-193/original/fold-{k}-detach/legacy-equivalence.json'));assert cert['passed'] and cert['state_max_error']==0
  ym=norm['y_mean'];ys=norm['y_std']
  for arm in ARMS:
   z=((a[arm][V]-ym)/ys).astype(np.float32);err=float(abs(z.astype(float)*ys+ym-a[arm][V]).max());assert err<1e-10
   q=z*ys+ym;assert q.dtype==np.float32;converted[arm][V]=q.astype(float);checks.append({'fold':k,'arm':arm,'normalized_output_recovery_max_error':err,'legacyprecision_minus_saved_maxabs':float(abs(q-a[arm][V]).max())})
 np.testing.assert_array_equal(converted['detach'],d['cnn']['seeds'][0]);assert all(np.isfinite(x).all() for x in converted.values());return converted,checks

def frozen(d):
 a=d['a'];comp=d['old']['components'].copy();comp[:,2]=d['cnn']['mean3'];w=np.array([0.,.1875,.1875,.125,.125]);w=w/w.sum();raw=comp@w;np.testing.assert_array_equal(raw,d['pipe']['raw']);base=np.empty(400);correction=np.empty(400);coef=np.empty(400);slopes=np.empty(400)
 for k in range(5):
  root=_repo_path(P, f'artifacts/spatial-joint-140/original/adjacent-fold-{k}');r=read(root/'report.json');cal=r['outer_calibrator'];assert cal==plan['calibrators'][str(k)]==read(_repo_path(P, f'artifacts/cnn-seed-mean-133/original/pipeline-fold-{k}/report.json'))['outer_calibrator'];z=load(root/'predictions.npz');assert sha(root/'predictions.npz')==r['predictions_sha256'];V=z['outer_indices'];np.testing.assert_array_equal(V,np.flatnonzero(a['fold']==k));coef[V]=.081*cal['slope'];slopes[V]=cal['slope'];assert plan['coefficient_by_fold'][str(k)]==.081*cal['slope'];base[V]=.9*(cal['target_center']+cal['slope']*(raw[V]-cal['prediction_center']))+.1*d['old']['short'][V];np.testing.assert_array_equal(base[V],z['base']);cor=[]
  for j in range(3):
   path=_repo_path(P, f'artifacts/bootstrap-selector-144/original/fold-{k}-block-{j}');rr=read(path/'report.json');zz=load(path/'predictions.npz');assert sha(path/'predictions.npz')==rr['predictions_sha256'];np.testing.assert_array_equal(zz['outer_indices'],V);np.testing.assert_array_equal(zz['weights']@zz['full_candidate_predictions'],zz['primary']);cor.append(zz['primary'])
  correction[V]=np.mean(cor,axis=0)
 p018=base+correction;p021=.9*p018+.1*d['b']['L_MLP'];np.testing.assert_array_equal(p018,d['agg']['primary']);np.testing.assert_array_equal(p021,a['best021']);return {'base':base,'correction':correction,'raw':raw,'coef':coef,'slope':slopes,'best021':p021}
def swap():
 q=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',q).groups();return int(float(v)*(1024**2 if u=='M' else 1024**3))
class Monitor:
 def __init__(self,name):self.name=name;self.start=time.perf_counter();self.stop=threading.Event();self.trace=[];self.initialswap=swap()
 def sample(self):
  q={'seconds':time.perf_counter()-self.start,'RSS':int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024,'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'swap':swap()};self.trace.append(q)
  if max(q['RSS'],q['peak_RSS'])>6e9 or q['swap']>self.initialswap or q['seconds']>300:write(_repo_path(R, self.name+'-STOP.json'),self.trace);os._exit(74)
 def loop(self):
  while not self.stop.wait(.5):self.sample()
 def __enter__(self):self.sample();self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start();return self
 def __exit__(self,*a):self.stop.set();self.worker.join();self.sample();write(_repo_path(R, self.name+'-resources.json'),{'trace':self.trace,'initial_swap':self.initialswap,'MPS_used':False})
