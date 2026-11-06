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
CURRENT_MODEL="convnext_tiny"
D=P/'experiments/cnn-optimizer-186'
def floor_stats(X,y,T,V,seed):
 mu=X[T].mean(0);var=X[T].var(0);rms=float(np.sqrt(var.mean()));scale=np.maximum(np.sqrt(var),rms) if rms>0 else np.ones_like(var)
 return {'feature_mean':torch.tensor(mu),'feature_var':torch.tensor(var),'feature_scale':torch.tensor(scale),'feature_RMS_floor':rms,'scaler_policy':'train_RMS_floor','target_mean':float(y[T].mean()),'target_std':float(y[T].std()),'train':T.tolist(),'validation':V.tolist(),'seed':seed,'dim':X.shape[1]}
old.stats=floor_stats

def ck(p):return torch.load(_repo_io(p),map_location='cpu',weights_only=True)
def save_head(path,X,y,T,V,seed):
 if path.exists():return ck(path)
 h,o=old.headfit(X,y,T,V,seed);torch.save(o,_repo_io(path));del h;return o

def rng_reference(seed):
 # No model fitting: reproduce only the known head-dropout RNG consumption.
 cpu=torch.get_rng_state().clone();mps=torch.mps.get_rng_state().clone();torch.manual_seed(seed+1000);rows=[];drop=nn.Dropout(.25).to('mps');drop.train();x=torch.ones((8,64),device='mps',requires_grad=True)
 for step in range(240):
  rows.append(c.rnghash());a=drop(x);b=drop(x);del a,b
 end=c.rnghash();torch.set_rng_state(cpu);torch.mps.set_rng_state(mps);return rows,end

def parameter_groups(h,tail,ratio):
 hp=list(h.parameters());tp=list(tail.parameters());parts=[hp,*[list(tail[0][i].parameters()) for i in range(3)],list(tail[2].parameters())]
 assert [id(p) for part in parts for p in part]==[id(p) for p in hp+tp];assert len(set(id(p) for p in hp+tp))==len(hp+tp)
 counts=[sum(p.numel() for p in part) for part in parts];assert counts[1:]==c.plan['tail_group_counts'];assert all(p.requires_grad for p in hp+tp)
 raw=[ratio**2,ratio,1.];den=sum(raw)/3;mult=[r/den for r in raw];lrs=[1e-4,*[1e-5*m for m in mult],1e-5];assert abs(sum(n*l for n,l in zip(counts[1:],lrs[1:]))/sum(counts[1:])-1e-5)<1e-19
 groups=[{'params':part,'lr':lr} for part,lr in zip(parts,lrs)];names=['head.'+n for n,_ in h.named_parameters()]+['tail.'+n for n,_ in tail.named_parameters()];mapping=[];offset=0
 for label,part,lr,reference in zip(c.plan['group_names'],parts,lrs,c.plan['decay_reference_LRs']):
  mapping.append({'name':label,'parameter_count':sum(p.numel() for p in part),'tensor_count':len(part),'parameters':names[offset:offset+len(part)],'initial_lr':lr,'decay_reference_initial_lr':reference});offset+=len(part)
 return groups,mapping

def decay_unit():
 # Equal effective shrinkage is independent of the gradient learning-rate groups.
 values=torch.tensor([1.,-2.,3.],device='mps');a=values.clone();b=values.clone();trace=[]
 for epoch in range(12):
  factor=.1+.9*(1+math.cos(math.pi*epoch/12))/2;decay=1-1e-5*factor*.01
  for step in range(20):a.mul_(decay);b.mul_(decay)
  assert torch.equal(a,b);trace.append({'epoch':epoch,'decay_factor':decay})
 return {'equal_shrinkage_240_steps':True,'trace':trace,'actual_loop_uses_reference_LR_not_group_update_LR':True}

def train(o,cache,X,y_train,recipe,seed,out,predict_indices,ratio,fold):
 train_tail=True;assert ratio in [1.,c.plan["ratio"]]
 started=time.monotonic();T=np.array(o['train']);assert len(y_train)==len(T);prefix,tail=c.model(CURRENT_MODEL);prefixhash=c.statehash(prefix);del prefix;tail.requires_grad_(train_tail)
 for m in tail.modules():
  if isinstance(m,nn.modules.batchnorm._BatchNorm):m.requires_grad_(False)
 initial=c.statehash(tail);bn={n:v.detach().clone() for n,v in tail.state_dict().items() if 'running_' in n or 'num_batches_tracked' in n};fixed={n:v.detach().clone() for n,v in tail.named_parameters() if not v.requires_grad};tail.eval().to('mps');h=old.head(X.shape[1]);h.load_state_dict(o['head']);assert c.statehash(h)==o['final_head_sha256'];initialhead=c.statehash(h);h.to('mps');mu,scale=old.tensors(o,'mps');yy=torch.tensor((y_train-o['target_mean'])/o['target_std'],dtype=torch.float32,device='mps');look={int(i):j for j,i in enumerate(T)};hp=list(h.parameters());tp=[p for p in tail.parameters() if p.requires_grad];groups,mapping=parameter_groups(h,tail,ratio);c.write(out/'parameter-groups.json',mapping);assert len(groups)==5;assert sum(p.numel() for p in tp)==14290944;assert len(list(tail.named_buffers()))==0 and not any(isinstance(mod,nn.modules.batchnorm._BatchNorm) for mod in tail.modules())
 opt=torch.optim.AdamW(groups,weight_decay=0.) if recipe['optimizer']=='AdamW' else torch.optim.SGD(groups,momentum=.9,dampening=0.,nesterov=False,weight_decay=0.)
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda ep:.1+.9*(1+math.cos(math.pi*ep/12))/2)
 reference_rng,reference_end=rng_reference(seed);rng=np.random.default_rng(seed+1);orders=[];updates=[];curve=[];rnglog=[];lrlog=[];decaylog=[];inputlog=[];counter=0;oldfolder=c.C/'convnext_tiny'/f'fold-{fold}'/'last_stage112';oldrec=c.read(oldfolder/'report.json');oldtrace=c.load(oldfolder/'updates.npz')['values'];guard=Guard(o,X,T,tail,out,mu,scale,train_tail);hook=tail.register_forward_hook(lambda module,inputs,output:guard.feature(output))
 def metric(epoch):
  guard.phase='train_metric_'+str(epoch);pp=old.predict(h,tail,cache,X,T,o,'mps');curve.append({'epoch':epoch,'train_RMSE':float(np.sqrt(np.mean((y_train-pp)**2)))})
 metric(0);assert curve[0]['train_RMSE']==oldrec['initial_train_RMSE'];torch.manual_seed(seed+1000);c.emit({'started_fold':fold,'ratio':ratio,'group_counts':[z['parameter_count'] for z in mapping]})
 for epoch in range(12):
  guard.phase='training';h.train();tail.eval();order=rng.permutation(T);orders.append(order);losses=[];assert all(not m.training for m in tail.modules());assert h.training
  for j in range(0,len(T),16):
   inds=order[j:j+16];state=c.rnghash();assert state==reference_rng[counter];rnglog.append(state);inputlog.append(inds.copy());opt.zero_grad(set_to_none=True)
   for start in range(0,len(inds),8):
    rows=inds[start:start+8];a=tail(torch.tensor(np.array(cache[rows]),device='mps'));prediction=h((a-mu)/scale).flatten();target=yy[[look[int(i)] for i in rows]];loss=F.smooth_l1_loss(prediction,target);assert torch.isfinite(loss);(loss*len(rows)/len(inds)).backward();losses.append(float(loss.detach()))
   norms=[]
   for params in [hp,tp]:norms.append(float(torch.stack([p.grad.detach().norm()**2 for p in params if p.grad is not None]).sum().sqrt()) if params else 0.)
   guard.gradient(norms,epoch,j//16);grad=float(nn.utils.clip_grad_norm_(hp+tp,1.));assert math.isfinite(grad);assert all(p.grad is None for p in tail.parameters() if not p.requires_grad)
   lrs=[g['lr'] for g in opt.param_groups];factor=.1+.9*(1+math.cos(math.pi*epoch/12))/2;expected=[m['initial_lr']*factor for m in mapping];np.testing.assert_array_equal(lrs,expected);assert all(g['weight_decay']==0 for g in opt.param_groups);assert abs(sum(m['parameter_count']*l for m,l in zip(mapping[1:],lrs[1:]))/14290944-1e-5*factor)<1e-19;decays=[1-ref*factor*.01 for ref in c.plan['decay_reference_LRs']];assert decays[1:]==[1-1e-5*factor*.01]*4;lrlog.append(lrs);decaylog.append(decays);before_optimizer_rng=c.rnghash()
   with torch.no_grad():
    for g,decay in zip(opt.param_groups,decays):
     for param in g['params']:param.mul_(decay)
   opt.step();assert c.rnghash()==before_optimizer_rng;row=[epoch,j//16,len(inds),lrs[0],1e-5*factor,grad,*norms];updates.append(row)
   np.testing.assert_array_equal(row[:5],oldtrace[counter,:5])
   if ratio==1.:np.testing.assert_array_equal(row,oldtrace[counter])
   counter+=1
  scheduler.step()
  if (epoch+1)%3==0:
   metric(epoch+1);curve[-1]['train_augmented_mean_loss']=float(np.mean(losses));c.emit({'fold':fold,'ratio':ratio,'epoch':epoch+1,'seconds':time.monotonic()-started})
  assert time.monotonic()-started<120
 assert all(int(opt.state[p]['step'])==240 for p in hp+tp);assert counter==240 and c.rnghash()==reference_end;assert len(updates)==12*math.ceil(len(T)/16);assert (c.statehash(tail)!=initial)==train_tail
 for n,v in tail.state_dict().items():
  if n in bn:assert torch.equal(v.cpu(),bn[n]),n
 for n,v in tail.named_parameters():
  if n in fixed:assert torch.equal(v.detach().cpu(),fixed[n]),n
 assert hashlib.sha256(np.stack(orders).tobytes()).hexdigest()==oldrec['order_sha256']
 guard.phase='validation';pred=old.predict(h,tail,cache,X,predict_indices,o,'mps')
 obj={**o,'head':{n:v.detach().cpu().clone() for n,v in h.state_dict().items()},'tail':{n:v.detach().cpu().clone() for n,v in tail.state_dict().items()},'head_initial':initialhead,'prefix_initial':prefixhash,'tail_initial':initial,'tail_final':c.statehash(tail),'final_head_sha256':c.statehash(h),'recipe':recipe,'adaptation_seed':seed,'order_adaptation_sha256':hashlib.sha256(np.stack(orders).tobytes()).hexdigest(),'update_count':len(updates),'curve':curve,'BN_and_prefix_frozen':True,'tail_trainable':train_tail,'model_name':CURRENT_MODEL,'tail_trainable_parameters':sum(p.numel() for p in tp),'optimizer_reset_from_warmhead':True,'ratio':ratio,'parameter_groups':mapping,'RNG_reference_exact_all240':True}
 torch.save(obj,_repo_io(out/'model.pt'));o2=ck(out/'model.pt');h.load_state_dict(o2['head']);tail.load_state_dict(o2['tail']);guard.phase='reload';again=old.predict(h,tail,cache,X,predict_indices,o2,'mps');np.testing.assert_array_equal(pred,again);guard.save(_repo_io(tail));hook.remove()
 c.arrays(out/'rng-and-groups.npz',order=np.stack(inputlog),group_LR=np.array(lrlog),decay_factor=np.array(decaylog),rng_CPU=np.array([q['CPU'] for q in rnglog]),rng_MPS=np.array([q['MPS'] for q in rnglog]));c.arrays(out/'predictions.npz',indices=predict_indices,prediction=pred);c.arrays(out/'updates.npz',values=np.array(updates),columns=['epoch','batch','rows','head_lr','tail_lr','preclip_total_gradnorm','preclip_head_gradnorm','preclip_tail_gradnorm'])
 rec={'all_optimizer_parameter_step_counters':240,'legacy_tail_lr_column':'uniform reference LR; actual five group LRs are in rng-and-groups.npz','ratio':ratio,'fold':fold,'RNG_reference_exact_all240':True,'group_LR_mean_and_schedule_exact_all240':True,'effective_decay_fixed_all240':True,'optimizer_has_no_RNG_side_effect_all240':True,'recipe':recipe,'tail_trainable':train_tail,'tail_trainable_parameters':sum(p.numel() for p in tp),'seconds':time.monotonic()-started,'updates':len(updates),'train_n':len(T),'prediction_n':len(predict_indices),'initial_train_RMSE':curve[0]['train_RMSE'],'final_train_RMSE':curve[-1]['train_RMSE'],'curve':curve,'gradient_norm_min':min(x[5] for x in updates),'gradient_norm_max':max(x[5] for x in updates),'gradient_clip_fraction':sum(x[5]>1 for x in updates)/len(updates),'head_lr_first_last':[updates[0][3],updates[-1][3]],'tail_lr_first_last':[updates[0][4],updates[-1][4]],'same_model_reload_exact':True,'BN_eval_affine_stats_unchanged':True,'nontrainable_gradients_none':True,'head_initial':initialhead,'order_sha256':obj['order_adaptation_sha256'],'model_sha256':c.sha(out/'model.pt'),'updates_sha256':c.sha(out/'updates.npz'),'predictions_sha256':c.sha(out/'predictions.npz')};
 if ratio==1.:
  reference=ck(oldfolder/'model.pt')
  for name in ['head','tail']:
   for n,value in obj[name].items():assert torch.equal(value,reference[name][n]),(name,n)
  np.testing.assert_array_equal(pred,c.load(oldfolder/'predictions.npz')['prediction']);assert curve==oldrec['curve'];assert c.read(out/'stability.json')==c.read(oldfolder/'stability.json');rec['existing190_control_exact']=True
 c.write(out/'report.json',rec);c.emit({'path':str(out.relative_to(R)),**{k:rec[k] for k in ['seconds','updates','initial_train_RMSE','final_train_RMSE','gradient_clip_fraction']}})
 del tail,h,opt,groups,hp,tp,obj,o2;gc.collect();torch.mps.empty_cache();return pred,rec
