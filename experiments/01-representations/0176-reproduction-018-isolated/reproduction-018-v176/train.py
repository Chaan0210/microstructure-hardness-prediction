"""Fresh018 full500 reproduction; no historic fitted learners loaded."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
import sys,json,time,hashlib,math,resource,socket,threading,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=_RepoPath(__file__).resolve().parents[3];OUT=_repo_path(ROOT, 'run');sys.path.insert(0,str(ROOT))
def no_network(*a,**kw):raise RuntimeError('Network disabled')
socket.create_connection=no_network;socket.socket.connect=no_network
import numpy as np,torch,joblib
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import base_train as b,mlp_head as h,residual_models as n
from feature_helpers import Short
threadpool_limits(2);torch.set_num_threads(2)
CNN_SEEDS=(20261003,20261023,20261043);HEAD_SEEDS=(20261128,20261205,20261219);KINDS=tuple(n.KINDS)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.pending');assert not p.exists()
 with tmp.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
 tmp.rename(p)
def emit(x):print(json.dumps(x),flush=True)
def arrays(p,**kw):
 with Path(p).open('xb') as f:np.savez_compressed(_repo_io(f),**kw)
def memory():return {'RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver':torch.mps.driver_allocated_memory()}
def cap():
 z=memory();assert z['RSS']<4e9 and z['MPS_driver']<2e9,z;return z
def input_data():
 meta=read(_repo_path(OUT, 'train-feature-plan.json'));v=read(_repo_path(OUT, 'train-feature-validation.json'));assert v['passed'] and sha(_repo_path(OUT, 'train-features.npz'))==v['feature_sha256']
 with np.load(_repo_path(OUT, 'train-features.npz'),allow_pickle=False) as z:d={name:z[name].copy() for name in z.files}
 assert d['ids'].tolist()==meta['IDs'] and len(d['ids'])==500
 schema=read(_repo_path(ROOT, 'config/schema.json'));d['Kblocks']=schema['compact_blocks'];d['blocks']=schema['physics_blocks']+['joint_token']*528
 assert len(d['selection'])==400 and d['ids'][d['selection']].tolist()==read(_repo_path(ROOT, 'config/development-ids.json'));return d


def prepare():
 d=input_data();recipe=read(_repo_path(ROOT, 'config/recipe.json'));weights=read(_repo_path(ROOT, 'licenses/current-model-provenance.json'))
 for w in weights['weights']:assert sha(_repo_path(ROOT, w['path']))==w['sha256']
 core=['train.py','base_train.py','cnn_model.py','compact_regression.py','mlp_head.py','feature_helpers.py','candidate_features.py','grain_features.py','latent_features.py','denoised_grain_features.py','residual_linear.py','residual_models.py','validate.py','extract.py'];files=[_repo_path(ROOT, name) for name in core]+list((_repo_path(ROOT, 'vendor')).rglob('*.py'))+list((_repo_path(ROOT, 'source-history')).glob('*.py'));sources={str(p.relative_to(ROOT)):sha(p) for p in files}
 plan={'study':176,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Reproduce historic018 full500 finaltraining; no subgroup holdout evaluation or new selection','features':'Fresh500rawfeatures; reuse official frozen pretrained DINO weights only. No fitted regression/calibration/neural/residual artifacts reused.','feature_sha256':sha(_repo_path(OUT, 'train-features.npz')),'feature_plan_sha256':sha(_repo_path(OUT, 'train-feature-plan.json')),'sources':sources,'config':{str(p.relative_to(ROOT)):sha(p) for p in (_repo_path(ROOT, 'config')).glob('*.json')},'ids':d['ids'].tolist(),'groups':d['g'].tolist(),'selection_ids':d['ids'][d['selection']].tolist(),'target_array_sha256':hashlib.sha256(d['y'].tobytes()).hexdigest(),'recipe':recipe,'tolerances':recipe['tolerances'],'fresh_counts':{'base_nodes':13,'CNN':39,'long_MLP':78,'short_B':4,'KRR_final':13,'KRR_inner':351,'affines':4,'residual_objects':60,'residual_nonnull':48},'test':'Only inference after training validation. Test CSV values ignored; fitting guard during inference.','resource':{'CPU_threads':2,'RSS_cap_bytes':4000000000,'MPS_driver_cap_bytes':2000000000,'estimated_train_and_validation_minutes':[18,30]},'root_query':'Root query=train500 only for reload/inference comparison. Never used for calibration or residual targets. No root in-sample performance reported as validation.'}
 write(_repo_path(OUT, 'plan.json'),plan)
 import importlib.metadata,platform
 write(_repo_path(OUT, 'environment.json'),{'python':sys.version,'machine':platform.machine(),'packages':{dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()},'torch_threads':torch.get_num_threads(),'MPS_available':torch.backends.mps.is_available()})
 emit({'PLAN_LOCKED':sha(_repo_path(OUT, 'plan.json'))})


def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for name,hsh in {**p['sources'],**p['config']}.items():assert sha(_repo_path(ROOT, name))==hsh,name
 assert sha(_repo_path(OUT, 'train-features.npz'))==p['feature_sha256'];return p


def filesystem_guard():
 touched=set()
 def audit(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).absolute()
  if p.is_relative_to(PROJECT) and not p.is_relative_to(ROOT) and not p.is_relative_to(_repo_path(PROJECT, '.venv')):
   raise RuntimeError('Training read/write outside standalone source+run: '+str(p))
  if p.is_relative_to(ROOT):touched.add(str(p.relative_to(ROOT)))
 sys.addaudithook(audit);return touched

def short_predict(m,B,Q,device='mps'):
 m.eval();out=[]
 with torch.no_grad():
  for st in range(0,len(Q),2):out.extend(m(torch.tensor(B[Q[st:st+2]],dtype=torch.float32,device=device)).cpu().numpy().astype(float))
 return np.array(out)
def short_fit(d,U,Q,offset,path):
 start=time.perf_counter();seed=20270215+offset;X=d['B'];y=d['y'];sc=StandardScaler().fit(X[U]);ym=float(y[U].mean());ys=float(y[U].std());m=Short(sc.mean_,sc.scale_,seed).to('mps');initial=h.state_hash(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.1);orders=np.random.default_rng(seed).permuted(np.tile(np.arange(len(U)),(12,1)),axis=1);seeds=[];curve=[]
 for ep,order in enumerate(orders):
  lr=1e-5+.5*(.0003-1e-5)*(1+math.cos(math.pi*ep/12));opt.param_groups[0]['lr']=lr;m.train();losses=[]
  for first in range(0,len(U),16):
   group=U[order[first:first+16]];opt.zero_grad(set_to_none=True)
   for off in range(0,len(group),2):
    ii=group[off:off+2];seedmicro=seed+10000*(ep+1)+first+off;torch.manual_seed(seedmicro);seeds.append(seedmicro);xx=torch.tensor(X[ii],dtype=torch.float32,device='mps');yy=torch.tensor((y[ii]-ym)/ys,dtype=torch.float32,device='mps');loss=F.smooth_l1_loss(m(xx),yy,beta=1.);assert torch.isfinite(loss);(loss*len(ii)/len(group)).backward();losses.append(float(loss.detach()))
   norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step()
  curve.append({'epoch':ep+1,'loss':float(np.mean(losses)),'lr':lr});cap()
 pred=short_predict(m,X,Q)*ys+ym;state={k:v.detach().cpu() for k,v in m.state_dict().items()};obj={'state':state,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'train':U.tolist(),'query':Q.tolist(),'seed':seed};torch.save(obj,_repo_io(path/'short.pt'));rec={'seed':seed,'initial_state':initial,'final_state':h.state_hash(state),'order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_sha256':hashlib.sha256(np.array(seeds,dtype=np.int64).tobytes()).hexdigest(),'curve':curve,'seconds':time.perf_counter()-start};del m,opt;torch.mps.empty_cache();return pred,rec

def node(d,U,Q,offset,path,with_short=False):
 if (path/'report.json').exists():
  r=read(path/'report.json');assert r['U']==U.tolist() and r['Q']==Q.tolist() and r['offset']==offset
  for f,hsh in r['files'].items():assert sha(path/f)==hsh
  with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:out={n:z[n].copy() for n in z.files}
  emit({'RESUME_175_NODE':str(path.relative_to(OUT))});return out
 path.mkdir(parents=True,exist_ok=False);start=time.perf_counter();root_fit=path.name=='outer' and len(U)==500 and np.array_equal(U,Q);assert root_fit or (not set(U)&set(Q) and not set(d['g'][U])&set(d['g'][Q]));pred={};logs={};ids=d['ids'].tolist();y=d['y']
 pred['KRR'],m,logs['KRR']=b.kernel_fold(d['K'],y,d['g'],d['Kblocks'],U,Q,offset,selection=d['selection'] if root_fit else U);joblib.dump({'model':m,'train':U,'query':Q},_repo_io(path/'kernel.joblib'));del m
 for seed in CNN_SEEDS:
  b.CNN_SEED=seed;pred[f'CNN_{seed}'],logs[f'CNN_{seed}']=b.cnn_fold(d['images'],y,U,Q,ids,offset,'mps',path/f'CNN-{seed}.pt');cap()
 pred['CNN']=np.mean([pred[f'CNN_{s}'] for s in CNN_SEEDS],axis=0)
 for arm in ['S','B']:
  X=d[arm];sc,ym,ys,edges,bins=h.preprocess(X[U],y[U]);xt=torch.tensor(sc.transform(X[U]),dtype=torch.float32);yt=torch.tensor((y[U]-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins)
  for seed in HEAD_SEEDS:
   m,logs[f'{arm}_{seed}']=h.fit(xt,yt,bt,seed+offset,0.);torch.save({'model':m.state_dict(),'input_dim':X.shape[1],'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'train':U.tolist(),'query':Q.tolist(),'seed':seed+offset},_repo_io(path/f'{arm}-{seed}.pt'));pred[f'{arm}_{seed}']=h.predict(m,sc.transform(X[Q]),ym,ys);del m
  pred[arm]=np.mean([pred[f'{arm}_{s}'] for s in HEAD_SEEDS],axis=0)
 if with_short:pred['short'],logs['short']=short_fit(d,U,Q,offset,path)
 pred['P']=np.column_stack([np.zeros(len(Q)),pred['KRR'],pred['CNN'],pred['S'],pred['B']]);arrays(path/'predictions.npz',U=U,Q=Q,**pred)
 r={'U':U.tolist(),'Q':Q.tolist(),'train_ids':d['ids'][U].tolist(),'query_ids':d['ids'][Q].tolist(),'offset':offset,'with_short':with_short,'KRR_selection':(d['selection'] if root_fit else U).tolist(),'root_in_sample_query':root_fit,'all_learners_fresh':True,'logs':logs,'seconds':time.perf_counter()-start,'memory':cap(),'files':{p.name:sha(p) for p in path.iterdir() if p.is_file()}};write(path/'report.json',r);emit({'NODE_COMPLETE':str(path.relative_to(OUT)),'seconds':r['seconds'],'memory':r['memory']});return pred

def mixture(P):
 w=np.array([0.,.1875,.1875,.125,.125]);return P@(w/w.sum())
def affine(P,y,Q):
 c=float(P.mean());m=float(y.mean());z=P-c;s=float(z@(y-m)/(z@z));return m+s*(Q-c),{'prediction_center':c,'target_center':m,'slope':s,'intercept':m-s*c}
def bootstrap(loss,g,seed):
 unique,inverse=np.unique(g,return_inverse=True);n=len(unique);draw=np.random.default_rng(seed).integers(0,n,(512,n));counts=np.stack([np.bincount(row,minlength=n) for row in draw]);wins=np.argmin(counts[:,inverse]@loss.T,axis=1);return np.bincount(wins,minlength=5)/512,draw,wins

def residual(d,B,anchor,V,k,j,path):
 path.mkdir();res=d['y'][B]-anchor;seed=20271013+100*k+j;boot=20271204+100*k+j;folds=list(GroupKFold(3,shuffle=True,random_state=seed).split(d['X'][B],res,d['g'][B]));ip=np.full((5,len(B)),np.nan)
 for t,(I,H) in enumerate(folds):
  assert not set(d['g'][B[I]])&set(d['g'][B[H]])
  for ci,kind in enumerate(KINDS):
   m=n.fit(d['X'][B[I]],res[I],d['blocks'],kind);ip[ci,H]=n.predict(m,d['X'][B[H]]);joblib.dump({'model':m,'train':B[I],'validation':B[H]},_repo_io(path/f'inner-{t}-{kind}.joblib'))
 w,draw,wins=bootstrap((ip-res)**2,d['g'][B],boot);full=[]
 for kind in KINDS:
  m=n.fit(d['X'][B],res,d['blocks'],kind);full.append(n.predict(m,d['X'][V]));joblib.dump({'model':m,'train':B},_repo_io(path/f'full-{kind}.joblib'))
 full=np.stack(full);pred=w@full;arrays(path/'predictions.npz',B=B,V=V,anchor=anchor,residual=res,inner=ip,weights=w,draws=draw,wins=wins,full=full,primary=pred);write(path/'report.json',{'foldseed':seed,'bootseed':boot,'splits':[{'train':B[I].tolist(),'validation':B[H].tolist()} for I,H in folds],'files':{p.name:sha(p) for p in path.iterdir() if p.is_file()}});return pred

def fit_full(d):
 root=_repo_path(OUT, 'full')
 if (root/'report.json').exists():return read(root/'report.json')
 start=time.perf_counter();T=np.arange(500);outer=node(d,T,T,0,root/'outer',True);middle=np.full((500,5),np.nan);cells=[]
 splits=list(GroupKFold(3,shuffle=True,random_state=20270110).split(d['S'],d['y'],d['g']))
 for j,(A,B) in enumerate(splits):
  offset=100*(j+1);mid=node(d,A,B,offset,root/f'middle-{j}',True);middle[B]=mid['P'];deep=np.full((len(A),5),np.nan)
  inner=list(GroupKFold(3,shuffle=True,random_state=20270110+offset).split(d['S'][A],d['y'][A],d['g'][A]))
  for ell,(C,D) in enumerate(inner):deep[D]=node(d,A[C],A[D],offset+100*(ell+1),root/f'deep-{j}-{ell}')['P']
  anchor,cal=affine(mixture(deep),d['y'][A],mixture(mid['P']));anchor=.9*anchor+.1*mid['short'];cells.append((A,B,anchor,cal,deep))
 assert np.isfinite(middle).all();cp,cal=affine(mixture(middle),d['y'],mixture(outer['P']));base=.9*cp+.1*outer['short'];cor=[]
 for j,(A,B,anchor,ac,deep) in enumerate(cells):
  cor.append(residual(d,B,anchor,T,0,j,root/f'residual-{j}'));arrays(root/f'anchor-{j}.npz',A=A,B=B,deep=deep,anchor=anchor)
 pred=base+np.mean(cor,0);assert np.isfinite(pred).all();arrays(root/'predictions.npz',T=T,V=T,middle=middle,outer=outer['P'],raw=mixture(outer['P']),short=outer['short'],base=base,corrections=np.stack(cor),primary=pred)
 r={'rows':500,'calibrator':cal,'anchor_calibrators':[v[3] for v in cells],'seconds':time.perf_counter()-start,'memory':cap(),'fresh_fit_counts':read(_repo_path(OUT, 'plan.json'))['fresh_counts'],'prediction_sha256':sha(root/'predictions.npz'),'node_reports':{str(p.relative_to(root)):sha(p) for p in root.glob('*/report.json')},'no_holdout_subgroup_metrics':True,'test_read':False};write(root/'report.json',r);emit({'FULL500_COMPLETE':{k:v for k,v in r.items() if k not in ['node_reports']}});return r


def run():
 check();d=input_data();lock=_repo_path(OUT, 'training.lock')
 with lock.open('x') as f:f.write(str(os.getpid()))
 touched=filesystem_guard();start=time.perf_counter();trace=[];stop=threading.Event()
 def watch():
  while not stop.wait(1):trace.append({'seconds':time.perf_counter()-start,**memory()})
 worker=threading.Thread(target=watch,daemon=True);worker.start()
 try:fit_full(d);assert time.perf_counter()-start<1800;cap()
 finally:
  stop.set();worker.join();write(_repo_path(OUT, f'execution-{os.getpid()}.json'),{'seconds':time.perf_counter()-start,'file_accesses':sorted(touched),'resource_trace':trace,'historical_project_files_blocked_by_audit_hook':True});lock.unlink()

if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='fit':run()
 else:raise ValueError('Use prepare or fit')
