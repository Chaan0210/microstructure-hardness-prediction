from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from transform import blur
from common import np,torch,R,P,time
F=torch.nn.functional

def initialize(k,dev=None):
 torch.manual_seed(20261003+k);m=c.legacy.TinyCNN().to(dev or c.device);assert sum(p.numel() for p in m.parameters())==91905;return m

def optimize(m):return torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)

def step(m,o,xb,yb,arm):
 m.train();o.zero_grad(set_to_none=True);pred=m(xb)
 loss=F.mse_loss(pred,yb)
 assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5.);assert torch.isfinite(gn);o.step()
 with torch.no_grad():
  mse=F.mse_loss(pred,yb);robust=2*F.huber_loss(pred,yb,delta=1.0);tail=(abs(pred-yb)>1).float().mean()
 return [float(loss.detach().cpu()),float(gn.detach().cpu()),float(mse.cpu()),float(robust.cpu()),float(tail.cpu())]

@torch.no_grad()
def predict(m,images,ix,norm,dev=None):
 dev=dev or c.device;m.eval();pred=[];zall=[]
 for j in range(0,len(ix),32):
  a=(images[ix[j:j+32]]-norm['x_mean'])/max(norm['x_std'],1e-6);z=m(torch.from_numpy(np.ascontiguousarray(a)).to(dev)).cpu().numpy();p=z*norm['y_std']+norm['y_mean'];assert z.dtype==p.dtype==np.float32;pred.extend(p.tolist());zall.extend(z.tolist())
 return np.array(pred),np.array(zall,dtype=np.float32)

def equal_tree(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal_tree(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal_tree(x,y)
 else:assert a==b

def fit(d,k,arm,folder):
 assert arm in ['control','blur'];folder.mkdir(exist_ok=False);start=time.perf_counter();T,V=c.indices(k);norm=c.norm(d,T);X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std']
 alpha=c.read(R/'preflight.json')['folds'][k]['alpha'];B=(blur(d['images'][T],alpha)-norm['x_mean'])/max(norm['x_std'],1e-6);phase=np.array([int.from_bytes(__import__('hashlib').sha256(('blur215:'+i).encode()).digest()[:8],'little')%2 for i in d['ids'][T]]);exposure=np.zeros((320,2),int)
 m=initialize(k);initial=c.statehash(m.state_dict());rngstart=c.rnghash();o=optimize(m);sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=80,eta_min=.0001);rng=np.random.default_rng(20261003+k);avg=None;orders=[];rots=[];flips=[];lrs=[];values=[];rngs=[];ih=[];eh={}
 c.emit({'fit_start':folder.name,'fold':k,'arm':arm,'loss_policy':'MSE'})
 for epoch in range(80):
  order=rng.permutation(320)
  for first in range(0,320,32):
   ix=order[first:first+32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);xb=X[ix].copy()
   if arm=='blur':
    selected=(epoch+phase[ix])%2==0;xb[selected]=B[ix[selected]];exposure[ix[selected],0]+=1;exposure[ix[~selected],1]+=1
   xb=np.rot90(xb,rot,axes=(2,3));xb=xb[:,:,:,::-1] if flip else xb;xb=np.ascontiguousarray(xb)
   ih.append(c.ah(xb));orders.append(ix.copy());rots.append(rot);flips.append(flip);lrs.append(o.param_groups[0]['lr']);rngs.append(c.rnghash());values.append(step(m,o,torch.from_numpy(xb).to(c.device),torch.tensor(y[ix],dtype=torch.float32,device=c.device),arm))
  sch.step()
  if epoch>=75:
   state={n:v.detach().cpu().clone() for n,v in m.state_dict().items()};eh[str(epoch+1)]=c.statehash(state);torch.save(state,_repo_io(folder/f'epoch-{epoch+1}.pt'))
   if avg is None:avg={n:v/5 for n,v in state.items()}
   else:
    for n,v in state.items():avg[n]+=v/5
  if (epoch+1)%20==0:c.emit({'fold':k,'arm':arm,'epoch':epoch+1,'seconds':time.perf_counter()-start})
  assert time.perf_counter()-start<180
 assert arm=='control' or np.array_equal(exposure,np.tile([40,40],(320,1)))
 rngend=c.rnghash();torch.save({'state':{n:v.detach().cpu().clone() for n,v in m.state_dict().items()},'optimizer':o.state_dict(),'scheduler':sch.state_dict()},_repo_io(folder/'final-live-and-optimizer.pt'))
 m.load_state_dict(avg);vp,zp=predict(m,d['images'],V,norm);tp,zt=predict(m,d['images'],T,norm);c.sync();assert np.isfinite(vp).all() and np.isfinite(tp).all()
 torch.save({'state_dict':avg,'normalization':norm,'train_indices':T.tolist(),'validation_indices':V.tolist(),'train_ids':d['ids'][T].tolist(),'validation_ids':d['ids'][V].tolist(),'arm':arm,'fold':k,'seed':20261003+k},_repo_io(folder/'model.pt'));c.arrays(folder/'predictions.npz',train_indices=T,validation_indices=V,train_predictions=tp,validation_predictions=vp,validation_normalized=zp,train_normalized=zt)
 c.arrays(folder/'streams.npz',order=np.stack(orders),rot=np.array(rots),flip=np.array(flips),lr=np.array(lrs),values=np.array(values),rng_CPU=np.array([a['CPU'] for a in rngs]),rng_MPS=np.array([a.get('MPS','CPU only') for a in rngs]),input_SHA256=np.array(ih),exposure=exposure)
 report={'fold':k,'arm':arm,'updates':len(orders),'normalization':norm,'initial_state_SHA256':initial,'rng_start':rngstart,'rng_end':rngend,'epoch_state_SHA256':eh,'averaged_state_SHA256':c.statehash(avg),'model_SHA256':c.sha(folder/'model.pt'),'predictions_SHA256':c.sha(folder/'predictions.npz'),'streams_SHA256':c.sha(folder/'streams.npz'),'final_live_optimizer_SHA256':c.sha(folder/'final-live-and-optimizer.pt'),'seconds':time.perf_counter()-start,'resource':c.resources(),'validation_score_computed':False,'stream_values_columns':['optimized_loss','unclipped_gradient_norm','unweighted_MSE','scaled_Huber','tail_fraction']};c.write(folder/'fit.json',report);c.emit({'fit_done':folder.name,'seconds':report['seconds']});del m,o,avg,state;__import__('gc').collect()
 if c.device=='mps':torch.mps.empty_cache()
 return vp,report
