from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,B,np,torch,nn
import importlib.util,time,math,gc,argparse,hashlib
from torch.nn import functional as F
from stability import Guard
spec=importlib.util.spec_from_file_location('base185',B/'train.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
RECIPES=[]
D=P/'experiments/cnn-optimizer-186'
def floor_stats(X,y,T,V,seed):
 mu=X[T].mean(0);var=X[T].var(0);rms=float(np.sqrt(var.mean()));scale=np.maximum(np.sqrt(var),rms) if rms>0 else np.ones_like(var)
 return {'feature_mean':torch.tensor(mu),'feature_var':torch.tensor(var),'feature_scale':torch.tensor(scale),'feature_RMS_floor':rms,'scaler_policy':'train_RMS_floor','target_mean':float(y[T].mean()),'target_std':float(y[T].std()),'train':T.tolist(),'validation':V.tolist(),'seed':seed,'dim':X.shape[1]}
old.stats=floor_stats

def ck(p):return torch.load(_repo_io(p),map_location='cpu',weights_only=True)
def save_head(path,X,y,T,V,seed):
 if path.exists():return ck(path)
 h,o=old.headfit(X,y,T,V,seed);torch.save(o,_repo_io(path));del h;return o

def decay_unit():
 a=nn.Parameter(torch.tensor([1.,-2.,3.]));b=nn.Parameter(a.detach().clone());one=torch.optim.AdamW([a],lr=.01,weight_decay=.01);two=torch.optim.AdamW([b],lr=.01,weight_decay=0.)
 for i in range(3):
  a.grad=torch.tensor([.1+i,-.2,.3]);b.grad=a.grad.clone()
  with torch.no_grad():b.mul_(1-.01*.01)
  one.step();two.step()
 assert torch.equal(a,b);return {'native_AdamW_vs_shared_explicit_decay_exact':True}

def train(o,cache,X,y_train,recipe,seed,out,predict_indices,train_tail=True):
 started=time.monotonic();T=np.array(o['train']);assert len(y_train)==len(T);prefix,tail=c.model(CURRENT_MODEL);prefixhash=c.statehash(prefix);del prefix;tail.requires_grad_(train_tail)
 for m in tail.modules():
  if isinstance(m,nn.modules.batchnorm._BatchNorm):m.requires_grad_(False)
 initial=c.statehash(tail);bn={n:v.detach().clone() for n,v in tail.state_dict().items() if 'running_' in n or 'num_batches_tracked' in n};fixed={n:v.detach().clone() for n,v in tail.named_parameters() if not v.requires_grad};tail.eval().to('mps');h=old.head(X.shape[1]);h.load_state_dict(o['head']);assert c.statehash(h)==o['final_head_sha256'];initialhead=c.statehash(h);h.to('mps');mu,scale=old.tensors(o,'mps');yy=torch.tensor((y_train-o['target_mean'])/o['target_std'],dtype=torch.float32,device='mps');look={int(i):j for j,i in enumerate(T)};hp=list(h.parameters());tp=[p for p in tail.parameters() if p.requires_grad];groups=[{'params':hp,'lr':recipe['head_lr']}]+([{'params':tp,'lr':recipe['tail_lr']}] if train_tail else [])
 opt=torch.optim.AdamW(groups,weight_decay=0.) if recipe['optimizer']=='AdamW' else torch.optim.SGD(groups,momentum=.9,dampening=0.,nesterov=False,weight_decay=0.)
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda ep:.1+.9*(1+math.cos(math.pi*ep/12))/2)
 rng=np.random.default_rng(seed+1);orders=[];updates=[];curve=[];guard=Guard(o,X,T,tail,out,mu,scale,train_tail);hook=tail.register_forward_hook(lambda module,inputs,output:guard.feature(output))
 def metric(epoch):
  guard.phase='train_metric_'+str(epoch);pp=old.predict(h,tail,cache,X,T,o,'mps');curve.append({'epoch':epoch,'train_RMSE':float(np.sqrt(np.mean((y_train-pp)**2)))})
 metric(0);torch.manual_seed(seed+1000)
 for epoch in range(12):
  guard.phase='training';h.train();tail.eval();order=rng.permutation(T);orders.append(order);losses=[];assert all(not m.training for m in tail.modules());assert h.training
  for j in range(0,len(T),16):
   inds=order[j:j+16];opt.zero_grad(set_to_none=True)
   for start in range(0,len(inds),8):
    rows=inds[start:start+8];a=tail(torch.tensor(np.array(cache[rows]),device='mps'));prediction=h((a-mu)/scale).flatten();target=yy[[look[int(i)] for i in rows]];loss=F.smooth_l1_loss(prediction,target);assert torch.isfinite(loss);(loss*len(rows)/len(inds)).backward();losses.append(float(loss.detach()))
   norms=[]
   for params in [hp,tp]:norms.append(float(torch.stack([p.grad.detach().norm()**2 for p in params if p.grad is not None]).sum().sqrt()) if params else 0.)
   guard.gradient(norms,epoch,j//16);grad=float(nn.utils.clip_grad_norm_(hp+tp,1.));assert math.isfinite(grad);assert all(p.grad is None for p in tail.parameters() if not p.requires_grad)
   lrs=[g['lr'] for g in opt.param_groups]+([] if train_tail else [0.])
   with torch.no_grad():
    for g in opt.param_groups:
     for param in g['params']:param.mul_(1-g['lr']*.01)
   opt.step();updates.append([epoch,j//16,len(inds),*lrs,grad,*norms])
  scheduler.step()
  if (epoch+1)%3==0:metric(epoch+1);curve[-1]['train_augmented_mean_loss']=float(np.mean(losses))
 assert len(updates)==12*math.ceil(len(T)/16);assert (c.statehash(tail)!=initial)==train_tail
 for n,v in tail.state_dict().items():
  if n in bn:assert torch.equal(v.cpu(),bn[n]),n
 for n,v in tail.named_parameters():
  if n in fixed:assert torch.equal(v.detach().cpu(),fixed[n]),n
 guard.phase='validation';pred=old.predict(h,tail,cache,X,predict_indices,o,'mps')
 obj={**o,'head':{n:v.detach().cpu().clone() for n,v in h.state_dict().items()},'tail':{n:v.detach().cpu().clone() for n,v in tail.state_dict().items()},'head_initial':initialhead,'prefix_initial':prefixhash,'tail_initial':initial,'tail_final':c.statehash(tail),'final_head_sha256':c.statehash(h),'recipe':recipe,'adaptation_seed':seed,'order_adaptation_sha256':hashlib.sha256(np.stack(orders).tobytes()).hexdigest(),'update_count':len(updates),'curve':curve,'BN_and_prefix_frozen':True,'tail_trainable':train_tail,'model_name':CURRENT_MODEL,'tail_trainable_parameters':sum(p.numel() for p in tp),'optimizer_reset_from_warmhead':True}
 torch.save(obj,_repo_io(out/'model.pt'));o2=ck(out/'model.pt');h.load_state_dict(o2['head']);tail.load_state_dict(o2['tail']);guard.phase='reload';again=old.predict(h,tail,cache,X,predict_indices,o2,'mps');np.testing.assert_array_equal(pred,again);guard.save(_repo_io(tail));hook.remove()
 c.arrays(out/'predictions.npz',indices=predict_indices,prediction=pred);c.arrays(out/'updates.npz',values=np.array(updates),columns=['epoch','batch','rows','head_lr','tail_lr','preclip_total_gradnorm','preclip_head_gradnorm','preclip_tail_gradnorm'])
 rec={'recipe':recipe,'tail_trainable':train_tail,'tail_trainable_parameters':sum(p.numel() for p in tp),'seconds':time.monotonic()-started,'updates':len(updates),'train_n':len(T),'prediction_n':len(predict_indices),'initial_train_RMSE':curve[0]['train_RMSE'],'final_train_RMSE':curve[-1]['train_RMSE'],'curve':curve,'gradient_norm_min':min(x[5] for x in updates),'gradient_norm_max':max(x[5] for x in updates),'gradient_clip_fraction':sum(x[5]>1 for x in updates)/len(updates),'head_lr_first_last':[updates[0][3],updates[-1][3]],'tail_lr_first_last':[updates[0][4],updates[-1][4]],'same_model_reload_exact':True,'BN_eval_affine_stats_unchanged':True,'nontrainable_gradients_none':True,'head_initial':initialhead,'order_sha256':obj['order_adaptation_sha256'],'model_sha256':c.sha(out/'model.pt'),'updates_sha256':c.sha(out/'updates.npz'),'predictions_sha256':c.sha(out/'predictions.npz')};c.write(out/'report.json',rec);c.emit({'path':str(out.relative_to(R)),**{k:rec[k] for k in ['seconds','updates','initial_train_RMSE','final_train_RMSE','gradient_clip_fraction']}})
 del tail,h,opt,groups,hp,tp,obj,o2;gc.collect();torch.mps.empty_cache();return pred,rec
