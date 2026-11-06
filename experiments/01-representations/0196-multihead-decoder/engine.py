from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import math,time,gc
import common as c
from common import np,torch,R
nn=torch.nn;F=nn.functional;cfg=c.plan['recipe']
class Mixer(nn.Module):
 def __init__(self):
  super().__init__();self.shared=nn.Linear(32,32,bias=False);self.out=nn.Linear(32,32);self.drop=nn.Dropout(.25);self.ff=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,32),nn.Dropout(.25))
 def mix(self,q,source,arm):
  B,L,_=q.shape;N=source.shape[1];Q=self.shared(q).reshape(B,L,4,8).transpose(1,2);KV=self.shared(source).reshape(B,N,4,8).transpose(1,2)
  if arm=='mha':weights=torch.softmax(Q@KV.transpose(-2,-1)/math.sqrt(8),dim=-1);context=weights@KV
  else:assert arm=='mean';weights=torch.full((B,4,L,N),1/N,device=q.device,dtype=q.dtype);context=KV.mean(2,keepdim=True).expand(B,4,L,8)
  return context.transpose(1,2).reshape(B,L,32),weights,Q,KV
 def forward(self,q,source,arm):
  h,w,_,_=self.mix(q,source,arm);q=q+self.drop(self.out(h));return q+self.ff(q),w
class Model(nn.Module):
 def __init__(self,norm):
  super().__init__();self.project=nn.Linear(770,32);self.token_norm=nn.LayerNorm(32);self.queries=nn.Parameter(torch.empty(4,32));nn.init.normal_(self.queries,std=.02);self.cross=Mixer();self.self_norm=nn.LayerNorm(32);self.self_mixer=Mixer();self.final_norm=nn.LayerNorm(32);self.head=nn.Sequential(nn.Linear(32,16),nn.GELU(),nn.Dropout(.25),nn.Linear(16,1));self.register_buffer('mean',torch.tensor(norm['mean'],dtype=torch.float32));self.register_buffer('scale',torch.tensor(norm['scale'],dtype=torch.float32));yy,xx=np.meshgrid(np.linspace(-1,1,36,dtype=np.float32),np.linspace(-1,1,36,dtype=np.float32),indexing='ij');self.register_buffer('coords',torch.from_numpy(np.stack((xx.ravel(),yy.ravel()),1)))
 def forward(self,x,arm,diagnostics=False,coords=None):
  pos=self.coords if coords is None else coords;z=torch.cat(((x-self.mean)/self.scale,pos.expand(len(x),-1,-1)),2);H=self.token_norm(F.gelu(self.project(z)));q=self.queries.expand(len(x),-1,-1);q,cross=self.cross(q,H,arm);z=self.self_norm(q);h,w,_,_=self.self_mixer.mix(z,z,arm);q=q+self.self_mixer.drop(self.self_mixer.out(h));q=q+self.self_mixer.ff(q);slots=self.final_norm(q);p=self.head(slots.mean(1)).squeeze(-1)
  if not diagnostics:return p
  return p,{'cross_weights':cross,'self_weights':w,'slots':slots}
def initialize(norm,k,device):
 torch.manual_seed(cfg['seed']+k);m=Model(norm).to(device);assert sum(p.numel() for p in m.parameters())==38209;return m
def snapshot(m):return {n:v.detach().cpu().clone() for n,v in m.state_dict().items()}
def seed(k,epoch,first):return cfg['seed']+k+10000*(epoch+1)+first
def lr(epoch):return cfg['min_lr']+.5*(cfg['lr']-cfg['min_lr'])*(1+math.cos(math.pi*epoch/cfg['epochs']))
def streams(k):return np.random.default_rng(cfg['seed']+k).permuted(np.tile(np.arange(320),(20,1)),axis=1)
def update(m,opt,reader,indices,y,norm,k,epoch,first,arm,collect=False):
 start=time.perf_counter();opt.zero_grad(set_to_none=True);loss_total=torch.zeros((),device='mps');input_hashes=[]
 for offset in range(0,len(indices),2):
  ii=indices[offset:offset+2];raw=reader.read(ii)
  if collect:input_hashes.append(c.ah(raw))
  torch.manual_seed(seed(k,epoch,first+offset));x=torch.from_numpy(raw).to('mps');target=torch.tensor((y[ii]-norm['target_mean'])/norm['target_std'],dtype=torch.float32,device='mps');p=m(x,arm);loss=F.smooth_l1_loss(p,target,beta=1);(loss*(len(ii)/len(indices))).backward();loss_total+=loss.detach()*(len(ii)/len(indices))
 grads=None
 if collect:
  grads={n:float(p.grad.detach().norm()) if p.grad is not None else None for n,p in m.named_parameters()};assert all(g is not None and np.isfinite(g) and g>0 for g in grads.values())
 gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn) and torch.isfinite(loss_total);opt.step();torch.mps.synchronize();return [float(loss_total.cpu()),float(gn.cpu()),opt.param_groups[0]['lr']],grads,input_hashes,time.perf_counter()-start
@torch.no_grad()
def predict(m,reader,ii,norm,arm,device,diagnostic=False):
 m.eval();out=[];ess=[];maxw=[];off=[];cos=[]
 for j in range(0,len(ii),2):
  x=torch.from_numpy(reader.read(ii[j:j+2])).to(device)
  if diagnostic:
   p,z=m(x,arm,True);w=z['cross_weights'];ess.extend((1/(w*w).sum(-1)).cpu().numpy().ravel().astype(float));maxw.extend(w.max(-1).values.cpu().numpy().ravel().astype(float));w=z['self_weights'];off.extend((1-w.diagonal(dim1=-2,dim2=-1)).cpu().numpy().ravel().astype(float));s=F.normalize(z['slots'],dim=-1);cc=s@s.transpose(-2,-1);mask=~torch.eye(4,dtype=torch.bool,device=device);cos.extend(cc[:,mask].cpu().numpy().ravel().astype(float))
  else:p=m(x,arm)
  out.extend(p.cpu().numpy().astype(float)*norm['target_std']+norm['target_mean'])
 p=np.array(out);assert np.isfinite(p).all()
 if diagnostic:return p,{'effective_patch_count_quantiles':np.quantile(ess,[0,.5,.95,1]).tolist(),'max_patchweight_quantiles':np.quantile(maxw,[0,.5,.95,1]).tolist(),'query_offdiagonal_mass_quantiles':np.quantile(off,[0,.5,.95,1]).tolist(),'slot_cosine_quantiles':np.quantile(cos,[0,.5,.95,1]).tolist()}
 return p
def fit(d,mom,k,arm,folder,max_updates=None):
 folder.mkdir(exist_ok=False);start=time.perf_counter();T,V=c.indices(d,k);norm=c.normalize(d,T,mom);V=T[:4] if max_updates else V;m=initialize(norm,k,'mps');initial=c.statehash(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.1);order=streams(k);used=[];seeds=[];trace=[];times=[];firstgrads=None;inputhash=None;updates=0
 c.emit({'fit_start':str(folder.relative_to(R)),'arm':arm,'fold':k,'updates':max_updates or 400})
 with c.TokenReader() as reader:
  for epoch in range(20):
   m.train()
   for group in opt.param_groups:group['lr']=lr(epoch)
   for j in range(0,320,16):
    ii=T[order[epoch,j:j+16]];values,grads,hashes,seconds=update(m,opt,reader,ii,d['y'],norm,k,epoch,j,arm,updates==0);trace.append(values);times.append(seconds);used.append(ii.copy());seeds.extend(seed(k,epoch,j+x) for x in range(0,16,2));updates+=1
    if updates==1:firstgrads=grads;inputhash=hashes
    assert time.perf_counter()-start<180
    if max_updates and updates==max_updates:break
   if not max_updates and (epoch+1)%5==0:c.emit({'fold':k,'arm':arm,'epoch':epoch+1,'seconds':time.perf_counter()-start,'online_train_loss':float(np.mean(np.array(trace)[-20:,0]))})
   if max_updates and updates==max_updates:break
  state=snapshot(m);vp,attention=predict(m,reader,V,norm,arm,'mps',True);tp=predict(m,reader,T,norm,arm,'mps')
  payload={'state':state,'target_mean':norm['target_mean'],'target_std':norm['target_std'],'arm':arm,'fold':k,'train_indices':T.tolist(),'validation_indices':V.tolist(),'train_ids':d['ids'][T].tolist(),'validation_ids':d['ids'][V].tolist(),'seed':cfg['seed']+k,'updates':updates};torch.save(payload,_repo_io(folder/'model.pt'));torch.save(opt.state_dict(),_repo_io(folder/'optimizer.pt'))
 c.arrays(folder/'normalization.npz',mean=norm['mean'],variance=norm['variance'],scale=norm['scale'],RMS=np.array(norm['RMS']),target_mean=np.array(norm['target_mean']),target_std=np.array(norm['target_std']));c.arrays(folder/'predictions.npz',indices=V,train_indices=T,ids=d['ids'][V],prediction=vp,train_prediction=tp);c.arrays(folder/'streams.npz',order=np.stack(used),dropout_seed=np.array(seeds,dtype=np.int64),trace=np.array(trace),update_seconds=np.array(times));report={'fold':k,'arm':arm,'parameters':38209,'updates':updates,'initial_state_SHA256':initial,'final_state_SHA256':c.statehash(state),'first_parameter_gradient_norms':firstgrads,'first_input_hashes':inputhash,'order_SHA256':c.ah(np.stack(used)),'dropseed_SHA256':c.ah(np.array(seeds,dtype=np.int64)),'LR_SHA256':c.ah(np.array(trace)[:,2]),'train_RMSE':c.rmse(d['y'][T],tp),'validation_RMSE':None if max_updates else c.rmse(d['y'][V],vp),'gradient_clip_fraction':float(np.mean(np.array(trace)[:,1]>1)),'attention_diagnostics':attention,'seconds':time.perf_counter()-start,'update_seconds_median_after2':float(np.median(times[2:])),'model_SHA256':c.sha(folder/'model.pt'),'optimizer_SHA256':c.sha(folder/'optimizer.pt'),'prediction_SHA256':c.sha(folder/'predictions.npz'),'streams_SHA256':c.sha(folder/'streams.npz')};c.write(folder/'report.json',report);c.emit({'fit_done':str(folder.relative_to(R)),'updates':updates,'seconds':report['seconds']});del m,opt,state,payload;gc.collect();torch.mps.empty_cache();return vp,report
