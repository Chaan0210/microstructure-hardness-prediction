from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import copy,gc
import common as c
from common import np,torch,time,R,P,cfg
F=torch.nn.functional
class Model(c.legacy.TinyCNN):
 def __init__(self,k):
  super().__init__();state=torch.get_rng_state().clone();mps=torch.mps.get_rng_state().clone();g=torch.Generator(device='cpu').manual_seed(cfg['aux_init_seed']+k)
  torch.set_rng_state(g.get_state());self.aux=torch.nn.Linear(194,4);torch.set_rng_state(state);assert torch.equal(torch.get_rng_state(),state) and torch.equal(torch.mps.get_rng_state(),mps)
 def forward(self,x,arm=None):
  if arm is None:return super().forward(x)
  h=self.features(x);stats=torch.cat([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt(),x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1)
  pred=self.head(stats).squeeze(1);logits=self.aux(stats.detach() if arm=='detach' else stats);return pred,logits
 def base_parameters(self):return [p for n,p in self.named_parameters() if not n.startswith('aux.')]
def rngstate():return {'cpu':torch.get_rng_state().clone(),'mps':torch.mps.get_rng_state().clone()}
def rnghash():return {k:c.ah(v.numpy()) for k,v in rngstate().items()}
def setrng(s):
 torch.set_rng_state(s['cpu']);torch.mps.set_rng_state(s['mps'])
def base_state(m):return {n:v.detach().cpu().clone() for n,v in m.state_dict().items() if not n.startswith('aux.')}
def initialize(k,device,plain=False):
 torch.manual_seed(cfg['base_seed']+k);m=(c.legacy.TinyCNN() if plain else Model(k)).to(device);assert sum(p.numel() for p in m.parameters())==(91905 if plain else 92685);return m
def optimizers(m,plain=False):
 main=torch.optim.AdamW(list(m.parameters()) if plain else m.base_parameters(),lr=.002,weight_decay=.01);aux=None if plain else torch.optim.AdamW(m.aux.parameters(),lr=.002,weight_decay=.01)
 return main,aux
def update(m,main,aux,xb,yb,bb,arm,collect=False):
 main.zero_grad(set_to_none=True)
 if aux is not None:aux.zero_grad(set_to_none=True)
 if aux is None:pred=m(xb);mse=F.mse_loss(pred,yb);ce=torch.zeros((),device=xb.device);mse.backward()
 else:
  pred,logits=m(xb,arm);mse=F.mse_loss(pred,yb);ce=F.cross_entropy(logits,bb);assert torch.isfinite(ce);mse.backward(retain_graph=True);(.3*ce).backward()
 assert torch.isfinite(mse)
 base=list(m.parameters()) if aux is None else m.base_parameters();pregrads={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if not n.startswith('aux.')} if collect else None
 gn=torch.nn.utils.clip_grad_norm_(base,5.);an=torch.nn.utils.clip_grad_norm_(m.aux.parameters(),5.) if aux is not None else torch.zeros((),device=xb.device);assert torch.isfinite(gn) and torch.isfinite(an)
 main.step()
 if aux is not None:aux.step()
 values=[float(v.detach().cpu()) for v in [mse,ce,gn,an]]
 return {'values':values,'prediction':pred.detach().cpu().clone() if collect else None,'gradients':pregrads}

def cpu_tree(z):
 if torch.is_tensor(z):return z.detach().cpu().clone()
 if isinstance(z,dict):return {k:cpu_tree(v) for k,v in z.items()}
 if isinstance(z,list):return [cpu_tree(v) for v in z]
 if isinstance(z,tuple):return tuple(cpu_tree(v) for v in z)
 return z

def fit(d,arm,folder):
 folder.mkdir(exist_ok=False);start=time.perf_counter();norm=c.normalization(d);edges=np.quantile(d['y'],[.25,.5,.75],method='linear');bins=np.searchsorted(edges,d['y'],side='right');assert np.all(np.diff(edges)>0) and np.all(np.bincount(bins,minlength=4)==125);X=(d['images']-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y']-norm['y_mean'])/norm['y_std'];m=initialize(0,'mps');m.train();initial=c.statehash(m.state_dict());initialbase=c.statehash(base_state(m));rngstart=rnghash();main,aux=optimizers(m);schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=80,eta_min=.0001) for o in [main,aux]];rng=np.random.default_rng(20261003);avg=None;orders=[];lengths=[];rots=[];flips=[];lrs=[];loss=[];state_hashes={};inputhash={};step=0
 c.emit({'fit_start':folder.name,'arm':arm,'train_rows':500,'epochs':80,'updates':1280})
 for epoch in range(80):
  order=rng.permutation(500)
  for first in range(0,500,32):
   ix=order[first:first+32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);xb=np.rot90(X[ix],rot,axes=(2,3));xb=xb[:,:,:,::-1] if flip else xb;xb=np.ascontiguousarray(xb)
   if step in [0,15,399,799,1279]:inputhash[str(step)]=c.ah(xb)
   row=np.full(32,-1,dtype=np.int64);row[:len(ix)]=ix;orders.append(row);lengths.append(len(ix));rots.append(rot);flips.append(flip);lrs.append(main.param_groups[0]['lr']);assert main.param_groups[0]['lr']==aux.param_groups[0]['lr'];q=update(m,main,aux,torch.from_numpy(xb).to('mps'),torch.tensor(y[ix],dtype=torch.float32,device='mps'),torch.tensor(bins[ix],dtype=torch.long,device='mps'),arm);loss.append(q['values']);step+=1
  for s in schedulers:s.step()
  if epoch>=75:
   state={n:v.detach().cpu().clone() for n,v in m.state_dict().items()};state_hashes[str(epoch+1)]=c.statehash(state);torch.save(state,_repo_io(folder/f'epoch-{epoch+1}.pt'))
   if avg is None:avg={n:v/5 for n,v in state.items()}
   else:
    for n,v in state.items():avg[n]+=v/5
  if (epoch+1)%20==0:c.emit({'fit':folder.name,'epoch':epoch+1,'updates':step,'seconds':time.perf_counter()-start})
  assert time.perf_counter()-start<240
 assert step==1280;final={'state':cpu_tree(m.state_dict()),'main_optimizer':cpu_tree(main.state_dict()),'aux_optimizer':cpu_tree(aux.state_dict()),'scheduler':[s.state_dict() for s in schedulers],'rng':rngstate()};torch.save(final,_repo_io(folder/'final-update-state.pt'));rngend=rnghash();obj={'state_dict':avg,'normalization':norm,'train_ids':d['ids'].tolist(),'bin_edges':edges.tolist(),'train_bins':bins.tolist(),'arm':arm,'seed':20261003,'aux_seed':202710193,'epochs':80,'updates':1280,'average_last_epochs':5,'training_plan_SHA256':c.sha(R/'plan.json')};torch.save(obj,_repo_io(folder/'model.pt'));del m,main,aux,avg,final;gc.collect();torch.mps.empty_cache();p=c.predict(obj,d['images'],'mps');c.arrays(folder/'predictions.npz',ids=d['ids'],prediction=p);c.arrays(folder/'streams.npz',order=np.stack(orders),length=np.array(lengths),rot=np.array(rots),flip=np.array(flips),lr=np.array(lrs),values=np.array(loss))
 rec={'arm':arm,'updates':step,'initial_state_SHA256':initial,'initial_base_SHA256':initialbase,'rng_start':rngstart,'rng_end':rngend,'input_hashes':inputhash,'epoch_state_hashes':state_hashes,'averaged_state_SHA256':c.statehash(obj['state_dict']),'model_SHA256':c.sha(folder/'model.pt'),'prediction_SHA256':c.sha(folder/'predictions.npz'),'streams_SHA256':c.sha(folder/'streams.npz'),'main_clipped_fraction':float(np.mean(np.array(loss)[:,2]>5)),'aux_clipped_fraction':float(np.mean(np.array(loss)[:,3]>5)),'seconds':time.perf_counter()-start,'no_holdout_subgroup_or_training_RMSE_report':True};c.write(folder/'report.json',rec);c.emit({'fit_done':folder.name,'updates':1280,'seconds':rec['seconds']});return obj,p
