from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,np,torch,nn
import argparse,time,gc,math
from torch.nn import functional as F

def head(dim):return nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,1))
def checkpoint(path):return torch.load(_repo_io(path),map_location='cpu',weights_only=True)
def featurecheck(name):
 z=c.read(R/name/'extraction.json');assert z['passed'];assert c.sha(R/name/'features.npz')==z['feature_sha256'];assert c.sha(R/name/'prefix.npy')==z['prefix_sha256']
 a=c.load(R/name/'features.npz');assert a['ids'].tolist()==c.plan['IDs'];return a['X'].astype(float)
def tensors(o,device):return (o['feature_mean'].to(device,dtype=torch.float32),o['feature_scale'].to(device,dtype=torch.float32))
def predict(h,tail,cache,X,inds,o,device):
 h.eval();mu,scale=tensors(o,device);out=[]
 if tail is not None:tail.eval()
 with torch.inference_mode():
  for i in range(0,len(inds),2 if tail is not None else 64):
   I=inds[i:i+(2 if tail is not None else 64)]
   a=tail(torch.tensor(np.array(cache[I]),device=device)) if tail is not None else torch.tensor(X[I],device=device,dtype=torch.float32)
   out.extend(h((a-mu)/scale).flatten().cpu().numpy().astype(float)*o['target_std']+o['target_mean'])
 a=np.array(out);assert np.isfinite(a).all();return a

def stats(X,y,T,V,seed):
 mu=X[T].mean(0);var=X[T].var(0);scale=np.sqrt(var);scale[scale==0]=1.
 return {'feature_mean':torch.tensor(mu),'feature_var':torch.tensor(var),'feature_scale':torch.tensor(scale),'target_mean':float(y[T].mean()),'target_std':float(y[T].std()),'train':T.tolist(),'validation':V.tolist(),'seed':seed,'dim':X.shape[1]}
def headfit(X,y,T,V,seed):
 o=stats(X,y,T,V,seed);torch.manual_seed(seed);h=head(X.shape[1]);o['initial_head_sha256']=c.statehash(h);mu,scale=tensors(o,'cpu');xx=(torch.tensor(X[T],dtype=torch.float32)-mu)/scale;yy=torch.tensor((y[T]-o['target_mean'])/o['target_std'],dtype=torch.float32)
 opt=torch.optim.AdamW(h.parameters(),lr=.0003,weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=.00001);rng=np.random.default_rng(seed+1);orders=[];curve=[];torch.manual_seed(seed+1000)
 for epoch in range(100):
  h.train();order=rng.permutation(len(T));orders.append(order);losses=[]
  for j in range(0,len(T),64):
   I=order[j:j+64];opt.zero_grad(set_to_none=True);loss=F.smooth_l1_loss(h(xx[I]).flatten(),yy[I]);loss.backward();nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  schedule.step()
  if (epoch+1)%20==0:curve.append({'epoch':epoch+1,'train_loss':float(np.mean(losses))})
 o['head']={k:v.detach().cpu().clone() for k,v in h.state_dict().items()};o['curve']=curve;o['order_sha256']=__import__('hashlib').sha256(np.stack(orders).tobytes()).hexdigest();o['final_head_sha256']=c.statehash(h)
 return h.eval(),o

def evaluate(d,pred):
 y=d['y'];ref=d['blend_L_MLP'];fold=d['fold'];rmse=lambda y,p:float(np.sqrt(np.mean((y-p)**2)));blend=.9*ref+.1*pred;fg=[rmse(y[fold==k],ref[fold==k])-rmse(y[fold==k],blend[fold==k]) for k in range(5)];unique,inv=np.unique(d['g'],return_inverse=True);rng=np.random.default_rng(202710185);boot=[]
 for _ in range(3000):
  counts=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv];boot.append(float(np.sqrt(counts@((y-ref)**2)/counts.sum())-np.sqrt(counts@((y-blend)**2)/counts.sum())))
 ci=np.quantile(boot,[.025,.975]).tolist();keep=np.argsort((y-ref)**2-(y-blend)**2)[:-5];gain=rmse(y,ref)-rmse(y,blend);dropgain=rmse(y[keep],ref[keep])-rmse(y[keep],blend[keep]);gate=gain>=.02 and sum(v>0 for v in fg)>=3 and ci[0]>0 and dropgain>0
 return {'RMSE':{'best021':rmse(y,ref),'standalone':rmse(y,pred),'blend10':rmse(y,blend)},'blend_gain':gain,'fold_blend_gains':fg,'positive_folds':sum(v>0 for v in fg),'conditional_group_bootstrap_CI95':ci,'gain_after_removing_top5':dropgain,'residual_correlation_vs021':float(np.corrcoef(y-pred,y-ref)[0,1]),'advance_gate':bool(gate),'uncertainty_note':'Conditional reused400; notselectionadjusted orindependenttest.'}

def assemble(name,split,arm):
 d=c.data(split);root=R/name/split;pred=np.full(400,np.nan);records=[]
 for k in range(5):
  path=root/f'fold-{k}';z=c.load(path/f'{arm}-prediction.npz');V=np.flatnonzero(d['fold']==k);np.testing.assert_array_equal(z['indices'],V);np.testing.assert_array_equal(z['ids'],d['ids'][V]);pred[V]=z['prediction'];records.append(c.read(path/f'{arm}-report.json'))
 assert np.isfinite(pred).all();report=evaluate(d,pred);report.update(model=name,split=split,arm=arm,all_reload_verified=True)
 if arm=='finetune112':
  control=c.read(root/'frozen112-report.json');report['gain_vs_matched_frozen112_blend']=control['RMSE']['blend10']-report['RMSE']['blend10'];report['advance_gate']=report['advance_gate'] and report['gain_vs_matched_frozen112_blend']>0
 c.arrays(root/f'{arm}-OOF.npz',ids=d['ids'],y=d['y'],groups=d['g'],fold=d['fold'],prediction=pred,best021=d['blend_L_MLP'],blend=.9*d['blend_L_MLP']+.1*pred)
 report['OOF_sha256']=c.sha(root/f'{arm}-OOF.npz');c.write(root/f'{arm}-report.json',report);c.emit(report)

def frozen(name,split):
 c.check();X=featurecheck(name);d=c.data(split);root=R/name/split;root.mkdir();start=time.monotonic()
 if split!='original':assert c.read(R/name/'original/validation.json')['passed'] and any(c.read(R/name/'original'/f'{a}-report.json')['advance_gate'] for a in ['frozen100','finetune112'])
 with c.Monitor(name+'-'+split+'-frozen',3600):
  for k in range(5):
   T=np.flatnonzero(d['fold']!=k);V=np.flatnonzero(d['fold']==k);path=root/f'fold-{k}';path.mkdir();h,o=headfit(X,d['y'],T,V,202710185+k);pp=predict(h,None,None,X,V,o,'cpu');trainpred=predict(h,None,None,X,T,o,'cpu');f=path/'frozen100.pt';torch.save(o,_repo_io(f));q=checkpoint(f);h2=head(X.shape[1]);h2.load_state_dict(q['head']);again=predict(h2,None,None,X,V,q,'cpu');np.testing.assert_array_equal(pp,again)
   c.arrays(path/'frozen100-prediction.npz',indices=V,ids=d['ids'][V],prediction=pp);c.write(path/'frozen100-report.json',{'fold':k,'checkpoint_sha256':c.sha(f),'reload_maxabs':0.,'train_RMSE':float(np.sqrt(np.mean((d['y'][T]-trainpred)**2))),'seconds_total':time.monotonic()-start});c.emit({'frozen':name,'split':split,'fold':k,'seconds':time.monotonic()-start})
  assemble(name,split,'frozen100')
 c.check()

def adapt_one(name,split,k,arm,X,d,cache):
 start=time.monotonic();path=R/name/split/f'fold-{k}';o=checkpoint(path/'frozen100.pt');T=np.array(o['train']);V=np.array(o['validation']);seed=o['seed'];prefix,tail=c.model(name);prefixhash=c.statehash(prefix);initialtail=c.statehash(tail);trainable=arm=='finetune112';tail.requires_grad_(trainable)
 for m in tail.modules():
  if isinstance(m,nn.modules.batchnorm._BatchNorm):m.requires_grad_(False)
 bn={n:v.detach().clone() for n,v in tail.state_dict().items() if 'running_' in n or 'num_batches_tracked' in n};bnparams={n:v.detach().clone() for n,v in tail.named_parameters() if not v.requires_grad};tail.eval().to('mps');torch.manual_seed(seed+2000);h=head(X.shape[1]);h.load_state_dict(o['head']);assert c.statehash(h)==o['final_head_sha256'];h.to('mps');mu,scale=tensors(o,'mps');y=torch.tensor((d['y']-o['target_mean'])/o['target_std'],device='mps',dtype=torch.float32);groups=[{'params':list(h.parameters()),'lr':.0003}]
 if trainable:groups.append({'params':[p for p in tail.parameters() if p.requires_grad],'lr':.00001})
 opt=torch.optim.AdamW(groups,weight_decay=.01);schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda ep:.1+.9*(1+math.cos(math.pi*ep/12))/2);rng=np.random.default_rng(seed+2001);orders=[];curve=[];torch.manual_seed(seed+3000)
 for epoch in range(12):
  h.train();tail.eval();order=rng.permutation(T);orders.append(order);losses=[]
  for j in range(0,len(T),16):
   I=order[j:j+16];opt.zero_grad(set_to_none=True)
   for b in range(0,len(I),2):
    B=I[b:b+2]
    a=tail(torch.tensor(np.array(cache[B]),device='mps')) if trainable else torch.tensor(X[B],dtype=torch.float32,device='mps')
    loss=F.smooth_l1_loss(h((a-mu)/scale).flatten(),y[B]);(loss*(len(B)/len(I))).backward();losses.append(float(loss.detach()))
   nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],1.);opt.step()
  schedule.step();curve.append({'epoch':epoch+1,'train_loss':float(np.mean(losses))})
  if (epoch+1)%3==0:c.emit({'model':name,'split':split,'fold':k,'arm':arm,'epoch':epoch+1,'seconds':time.monotonic()-start})
 assert c.statehash(prefix)==prefixhash
 for n,v in tail.state_dict().items():
  if n in bn:assert torch.equal(v.cpu(),bn[n]),n
 for n,v in tail.named_parameters():
  if n in bnparams:assert torch.equal(v.detach().cpu(),bnparams[n]),n
 assert (c.statehash(tail)!=initialtail)==trainable
 pp=predict(h,tail if trainable else None,cache,X,V,o,'mps');trainpred=predict(h,tail if trainable else None,cache,X,T,o,'mps')
 obj={**o,'head':{n:v.detach().cpu().clone() for n,v in h.state_dict().items()},'tail':{n:v.detach().cpu().clone() for n,v in tail.state_dict().items()} if trainable else {},'arm':arm,'prefix_initial_sha256':prefixhash,'tail_initial_sha256':initialtail,'tail_final_sha256':c.statehash(tail),'head_initial_for_adaptation_sha256':o['final_head_sha256'],'order_adaptation_sha256':__import__('hashlib').sha256(np.stack(orders).tobytes()).hexdigest(),'adaptation_curve':curve,'BN_unchanged':True,'final_head_sha256':c.statehash(h)}
 f=path/f'{arm}.pt';torch.save(obj,_repo_io(f));q=checkpoint(f);h2=head(X.shape[1]);h2.load_state_dict(q['head']);h2.to('mps')
 if trainable:tail.load_state_dict(q['tail'])
 again=predict(h2,tail if trainable else None,cache,X,V,q,'mps');np.testing.assert_array_equal(pp,again)
 c.arrays(path/f'{arm}-prediction.npz',indices=V,ids=d['ids'][V],prediction=pp);rec={'fold':k,'arm':arm,'seconds':time.monotonic()-start,'checkpoint_sha256':c.sha(f),'reload_maxabs':0.,'train_RMSE':float(np.sqrt(np.mean((d['y'][T]-trainpred)**2))),'BN_and_prefix_unchanged':True,'trainable_tail_parameters':sum(p.numel() for p in tail.parameters() if p.requires_grad)};c.write(path/f'{arm}-report.json',rec);c.emit(rec);del prefix,tail,h,h2,opt,groups;gc.collect();torch.mps.empty_cache()

def adapt(name,split):
 c.check();X=featurecheck(name);d=c.data(split);cache=np.load(_repo_io(R/name/'prefix.npy'),mmap_mode='r')
 with c.Monitor(name+'-'+split+'-adapt',3600):
  for k in range(5):
   for arm in ['frozen112','finetune112']:adapt_one(name,split,k,arm,X,d,cache)
  for arm in ['frozen112','finetune112']:assemble(name,split,arm)
 c.check()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['frozen','adapt']);ap.add_argument('model',choices=['resnet50','efficientnet_b0']);ap.add_argument('split',choices=['original','20261031','20261101']);a=ap.parse_args();assert c.sha(R/'train.py')==c.read(R/'train-code-lock.json')['train.py'];frozen(a.model,a.split) if a.mode=='frozen' else adapt(a.model,a.split)
