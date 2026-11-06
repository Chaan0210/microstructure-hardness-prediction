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
@torch.no_grad()
def predict(m,images,ix,norm,device):
 m.eval();out=[]
 for j in range(0,len(ix),64):
  x=np.ascontiguousarray((images[ix[j:j+64]]-norm['x_mean'])/max(norm['x_std'],1e-6));out.extend(m(torch.from_numpy(x).to(device)).cpu().numpy().astype(float)*norm['y_std']+norm['y_mean'])
 return np.array(out)
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
def fit(d,k,arm,folder):
 folder.mkdir(exist_ok=False);start=time.perf_counter();T,V=c.indices(d,k);norm,edges,bins,perm,shuffled=c.preprocessing(d,T,k);X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];labels=shuffled if arm=='permuted' else bins
 m=initialize(k,'mps');m.train();initial=c.statehash(m.state_dict());initialbase=c.statehash(base_state(m));rngstart=rnghash();main,aux=optimizers(m);sched=[torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=80,eta_min=.0001) for o in [main,aux]];rng=np.random.default_rng(cfg['base_seed']+k);avg=None;orders=[];rots=[];flips=[];lrs=[];loss=[];state_hashes={};inputhash={};step=0
 c.emit({'fit_start':str(folder.relative_to(R)),'arm':arm,'fold':k})
 for epoch in range(80):
  order=rng.permutation(320)
  for first in range(0,320,32):
   ix=order[first:first+32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);xb=np.rot90(X[ix],rot,axes=(2,3));xb=xb[:,:,:,::-1] if flip else xb;xb=np.ascontiguousarray(xb)
   if step in [0,99,399,799]:inputhash[str(step)]=c.ah(xb)
   orders.append(ix.copy());rots.append(rot);flips.append(flip);lrs.append(main.param_groups[0]['lr']);assert main.param_groups[0]['lr']==aux.param_groups[0]['lr'];ret=update(m,main,aux,torch.from_numpy(xb).to('mps'),torch.tensor(y[ix],dtype=torch.float32,device='mps'),torch.tensor(labels[ix],dtype=torch.long,device='mps'),arm);loss.append(ret['values']);step+=1
  for s in sched:s.step()
  if epoch>=75:
   state={n:v.detach().cpu().clone() for n,v in m.state_dict().items()};state_hashes[str(epoch+1)]=c.statehash(state);torch.save(state,_repo_io(folder/f'epoch-{epoch+1}.pt'))
   if avg is None:avg={n:v/5 for n,v in state.items()}
   else:
    for n,v in state.items():avg[n]+=v/5
  if (epoch+1)%20==0:c.emit({'fold':k,'arm':arm,'epoch':epoch+1,'seconds':time.perf_counter()-start})
  assert time.perf_counter()-start<180
 rngend=rnghash();m.load_state_dict(avg);vp=predict(m,d['images'],V,norm,'mps');tp=predict(m,d['images'],T,norm,'mps');torch.mps.synchronize();assert np.isfinite(vp).all() and np.isfinite(tp).all()
 torch.save({'state_dict':avg,'normalization':norm,'train_indices':T.tolist(),'validation_indices':V.tolist(),'train_ids':d['ids'][T].tolist(),'validation_ids':d['ids'][V].tolist(),'bin_edges':edges.tolist(),'train_bins':bins.tolist(),'permutation':perm.tolist(),'used_bins':labels.tolist(),'arm':arm,'fold':k,'seed':cfg['base_seed']+k},_repo_io(folder/'model.pt'));c.arrays(folder/'predictions.npz',train_indices=T,validation_indices=V,train_predictions=tp,validation_predictions=vp);c.arrays(folder/'streams.npz',order=np.stack(orders),rot=np.array(rots),flip=np.array(flips),lr=np.array(lrs),values=np.array(loss))
 report={'arm':arm,'fold':k,'updates':step,'train_RMSE':c.rmse(d['y'][T],tp),'validation_RMSE':c.rmse(d['y'][V],vp),'initial_state_SHA256':initial,'initial_base_SHA256':initialbase,'rng_start':rngstart,'rng_end':rngend,'input_hashes':inputhash,'epoch_state_hashes':state_hashes,'averaged_state_SHA256':c.statehash(avg),'model_SHA256':c.sha(folder/'model.pt'),'predictions_SHA256':c.sha(folder/'predictions.npz'),'streams_SHA256':c.sha(folder/'streams.npz'),'main_clipped_fraction':float(np.mean(np.array(loss)[:,2]>5)),'aux_clipped_fraction':float(np.mean(np.array(loss)[:,3]>5)),'seconds':time.perf_counter()-start,'resource':c.resources()};c.write(folder/'report.json',report);c.emit({'fit_done':folder.name,'seconds':report['seconds'],'train_RMSE':report['train_RMSE'],'validation_RMSE':report['validation_RMSE']});del m,main,aux,avg;gc.collect();torch.mps.empty_cache();return vp,report
