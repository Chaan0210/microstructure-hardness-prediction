"""Concrete KRR/MLP/CNN backend. Main training remains hard-disabled for219 implementation stage."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from dataclasses import dataclass
from pathlib import Path
import json,math,joblib,hashlib,resource
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from contracts import require,sha
from compact_regression import BalancedKernel
from cnn_model import TinyCNN
from short_model import Short
import mlp_head as h
MAIN_TRAINING_ENABLED=False
CNN_SEEDS=(20261003,20261023,20261043)
HEAD_SEEDS=(20261128,20261205,20261219)
def training_guard():
 if not MAIN_TRAINING_ENABLED:raise RuntimeError('219 is implementation/CPU-unit-test only; main training is not enabled')
def resource_check(device):
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;gpu=torch.mps.driver_allocated_memory() if device=='mps' else 0
 require(rss+gpu<6000000000,'D2 memory cap exceeded')
 return {'RSS_peak_bytes':rss,'MPS_driver_bytes':gpu}
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
@dataclass
class NeuralPredictor:
 kind:str
 state:dict
 normalization:dict
 train:tuple
 seed:int
 anchor_sha256:str=''
 def predict(self,f,Q):
  n=self.normalization
  if self.kind=='CNN':
   model=TinyCNN().eval();model.load_state_dict(self.state);X=f.take('images',Q);batch=32
  elif self.kind=='short':
   model=Short(n['mean'],n['scale'],self.seed).eval();model.load_state_dict(self.state);X=f.take('B',Q);batch=2
  else:
   model=h.Head(len(n['mean'])).eval();model.load_state_dict(self.state);X=f.take(self.kind,Q);batch=64
  out=[]
  with torch.inference_mode():
   for start in range(0,len(X),batch):
    x=X[start:start+batch]
    if self.kind=='CNN':x=(x-n['x_mean'])/max(n['x_std'],1e-6)
    elif self.kind!='short':x=(x-n['mean'])/n['scale']
    p=model(torch.tensor(x,dtype=torch.float32));p=p[0] if self.kind not in ('short','CNN') else p
    # Preserve old CNN float32 inverse-normalization semantics.
    if self.kind=='CNN':out.extend((p.numpy()*n['y_std']+n['y_mean']).tolist())
    else:out.extend((p.numpy().astype(float)*n['y_std']+n['y_mean']).tolist())
  return np.array(out)
@dataclass
class Average:
 models:list
 train:tuple
 def predict(self,f,Q):return np.mean([m.predict(f,Q) for m in self.models],axis=0)
@dataclass
class BaseNode:
 kernel:object
 cnn:object
 S:object
 B:object
 train:tuple
 def predict(self,f,Q):return .3*self.kernel.predict(f.take('K',Q))+.3*self.cnn.predict(f,Q)+.2*self.S.predict(f,Q)+.2*self.B.predict(f,Q)

def cnn_normalization(X,y,residual=False,target_stats=None):
 mu,sd=float(X.mean()),float(X.std());ym,ys=float(y.mean()),float(y.std());require(sd>0 and (residual or ys>0),'Degenerate normalization')
 if residual:
  require(target_stats is not None,'Residual arms require shared true-B normalization');ym,ys=map(float,target_stats);require(np.isfinite([ym,ys]).all() and ys>=1e-6,'Invalid shared residual normalization')
 return mu,sd,ym,ys

def fit_cnn(f,labels,seed,device='mps',residual=False,anchor_sha256='',target_stats=None):
 training_guard();U=labels.ids;X=f.take('images',U);y=labels.take(U);mu,sd,ym,ys=cnn_normalization(X,y,residual,target_stats)
 Xt=(X-mu)/max(sd,1e-6);yt=(y-ym)/ys;rng=np.random.default_rng(seed);torch.manual_seed(seed);model=TinyCNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=.0001);avg=None;curve=[]
 for epoch in range(80):
  model.train();order=rng.permutation(len(U));loss_sum=0.
  for first in range(0,len(U),32):
   ix=order[first:first+32];xb=np.rot90(Xt[ix],int(rng.integers(4)),axes=(2,3))
   if rng.random()<.5:xb=xb[:,:,:,::-1]
   xx=torch.from_numpy(np.ascontiguousarray(xb)).to(device);target=torch.tensor(yt[ix],dtype=torch.float32,device=device);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(xx),target);require(bool(torch.isfinite(loss)),'CNN nonfinite loss');loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);require(bool(torch.isfinite(norm)),'CNN nonfinite gradient');opt.step();loss_sum+=float(loss.detach().cpu())*len(ix)
  scheduler.step();curve.append({'epoch':epoch+1,'train_rmse_augmented':float(np.sqrt(loss_sum/len(U))*ys),'memory':resource_check(device)})
  if epoch>=75:
   state=cpu_state(model)
   if avg is None:avg={k:v/5 for k,v in state.items()}
   else:
    for k in avg:avg[k]+=state[k]/5
 require(all(torch.isfinite(x).all() for x in avg.values()),'CNN invalid state');out=NeuralPredictor('CNN',avg,{'x_mean':mu,'x_std':sd,'y_mean':ym,'y_std':ys},tuple(U),seed,anchor_sha256);del model,opt
 if device=='mps':torch.mps.empty_cache()
 return out,curve

class ProductionBackend:
 def __init__(self,root,registry,device='mps',artifact_dir=None):
  self.root=Path(root);self.registry=registry;self.device=device;self.events=[];self.artifact_dir=Path(artifact_dir) if artifact_dir is not None else None
 def retain(self,kind,labels,offset,model,record):
  require(self.artifact_dir is not None,'Fresh D2 training requires a separate artifact directory')
  key=hashlib.sha256(labels.ids.tobytes()).hexdigest()[:16];p=self.artifact_dir/f'{kind}-{offset}-{key}.joblib';p.parent.mkdir(parents=True,exist_ok=True);require(not p.exists(),'Refuse model artifact overwrite');joblib.dump({'model':model,'record':record,'train':labels.ids.tolist(),'target_sha256':hashlib.sha256(labels.take(labels.ids).tobytes()).hexdigest()},_repo_io(p));self.events.append({'artifact':str(p),'sha256':sha(p),'kind':kind,'offset':offset,'train':labels.ids.tolist()})
 def entry(self,labels,offset):
  entries=[e for e in self.registry if e['train']==labels.ids.tolist() and e['offset']==offset];require(len(entries)<=1,'Ambiguous cache')
  if entries:require(hashlib.sha256(labels.take(labels.ids).tobytes()).hexdigest()==entries[0]['target_sha256'],'Cached target membership/version mismatch')
  return entries[0] if entries else None
 def checked(self,relative,digest):
  p=self.root/relative;require(sha(p)==digest,'Cached model checksum mismatch: '+relative);return p
 def obj(self,entry,key,labels,f):
  item=entry['files'][key];p=self.checked(item['path'],item['sha256']);obj=joblib.load(_repo_io(p)) if p.suffix=='.joblib' else torch.load(_repo_io(p),map_location='cpu',weights_only=False)
  require(obj['train_ids']==f.ids[labels.ids].tolist(),'Cached model training IDs mismatch');return obj
 def cached_base(self,f,labels,entry):
  U=tuple(labels.ids);kernel=self.obj(entry,'kernel',labels,f)['model'];cnns=[]
  for seed in CNN_SEEDS:
   obj=self.obj(entry,f'CNN-{seed}',labels,f);require(obj['seed']==seed+entry['offset'],'CNN seed mismatch');cnns.append(NeuralPredictor('CNN',obj['state_dict'],obj['normalization'],U,obj['seed']))
  arms={}
  for arm in ('S','B'):
   models=[]
   for seed in HEAD_SEEDS:
    obj=self.obj(entry,f'{arm}-{seed}',labels,f);require(obj['seed']==seed+entry['offset'],'Head seed mismatch');models.append(NeuralPredictor(arm,obj['model'],{'mean':np.asarray(obj['scaler_mean']),'scale':np.asarray(obj['scaler_scale']),'y_mean':obj['target_mean'],'y_std':obj['target_std']},U,obj['seed']))
   arms[arm]=Average(models,U)
  return BaseNode(kernel,Average(cnns,U),arms['S'],arms['B'],U)
 def base(self,f,labels,offset):
  entry=self.entry(labels,offset)
  if entry is not None:return self.cached_base(f,labels,entry)
  training_guard();require(self.artifact_dir is not None,'Fresh D2 training requires artifacts');U=labels.ids;y=labels.take(U);X=f.take('K',U);g=f.groups[U];folds=list(GroupKFold(3,shuffle=True,random_state=20261005+offset).split(U,groups=g));scores=[]
  for width in (.25,1.,4.):
   for alpha in (.01,.1,1.):
    p=np.full(len(U),np.nan)
    for ti,vi in folds:
     f.partition(U,U[ti],U[vi]);m=BalancedKernel(f.kblocks,width,alpha).fit(X[ti],y[ti]);p[vi]=m.predict(X[vi])
    scores.append((float(np.mean((p-y)**2)),width,alpha))
  best=min(range(len(scores)),key=lambda i:scores[i][0]);_,width,alpha=scores[best];kernel=BalancedKernel(f.kblocks,width,alpha).fit(X,y);cnn_models=[];curves=[]
  for seed in CNN_SEEDS:
   m,curve=fit_cnn(f,labels,seed+offset,self.device);cnn_models.append(m);curves.append(curve)
  cnn=Average(cnn_models,tuple(U));model=BaseNode(kernel,cnn,self.long(f,labels,offset,'S'),self.long(f,labels,offset,'B'),tuple(U));self.retain('base',labels,offset,model,{'KRR_selection_scores':scores,'CNN_curves':curves});return model
 def long(self,f,labels,offset,arm):
  training_guard();require(self.artifact_dir is not None,'Fresh D2 training requires artifacts');U=labels.ids;X=f.take(arm,U);y=labels.take(U);sc,ym,ys,edges,bins=h.preprocess(X,y);xt=torch.tensor(sc.transform(X),dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins);models=[]
  for seed in HEAD_SEEDS:
   m,record=h.fit(xt,yt,bt,seed+offset,0.,validation=None);models.append(NeuralPredictor(arm,cpu_state(m),{'mean':sc.mean_,'scale':sc.scale_,'y_mean':ym,'y_std':ys},tuple(U),seed+offset));self.events.append({'kind':arm,'train':U.tolist(),'seed':seed+offset,'record':record})
  out=Average(models,tuple(U));self.retain('long-'+arm,labels,offset,out,{'seed_count':3});return out
 def short(self,f,labels,offset):
  U=labels.ids;entry=self.entry(labels,offset)
  if entry is not None and 'short' in entry['files']:
   obj=self.obj(entry,'short',labels,f);require(obj['seed']==20270215+offset,'Short seed');return NeuralPredictor('short',obj['state'],{'mean':np.asarray(obj['mean']),'scale':np.asarray(obj['scale']),'y_mean':obj['target_mean'],'y_std':obj['target_std']},tuple(U),obj['seed'])
  training_guard();require(self.artifact_dir is not None,'Fresh D2 training requires artifacts');X=f.take('B',U);y=labels.take(U);seed=20270215+offset;sc=StandardScaler().fit(X);ym,ys=float(y.mean()),float(y.std());require(ys>0,'Degenerate short target');m=Short(sc.mean_,sc.scale_,seed).to(self.device);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.1);orders=np.random.default_rng(seed).permuted(np.tile(np.arange(len(U)),(12,1)),axis=1)
  for ep,order in enumerate(orders):
   opt.param_groups[0]['lr']=1e-5+.5*(.0003-1e-5)*(1+math.cos(math.pi*ep/12));m.train()
   for first in range(0,len(U),16):
    group=order[first:first+16];opt.zero_grad(set_to_none=True)
    for off in range(0,len(group),2):
     ii=group[off:off+2];torch.manual_seed(seed+10000*(ep+1)+first+off);xx=torch.tensor(X[ii],dtype=torch.float32,device=self.device);yy=torch.tensor((y[ii]-ym)/ys,dtype=torch.float32,device=self.device);loss=torch.nn.functional.smooth_l1_loss(m(xx),yy,beta=1.);require(bool(torch.isfinite(loss)),'Short nonfinite loss');(loss*len(ii)/len(group)).backward()
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);require(bool(torch.isfinite(norm)),'Short nonfinite gradient');opt.step()
  out=NeuralPredictor('short',cpu_state(m),{'mean':sc.mean_,'scale':sc.scale_,'y_mean':ym,'y_std':ys},tuple(U),seed);del m,opt
  if self.device=='mps':torch.mps.empty_cache()
  self.retain('short',labels,offset,out,{'epochs':12});return out
