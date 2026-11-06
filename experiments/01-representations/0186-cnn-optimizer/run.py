from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,B,np,torch,nn
import importlib.util,time,math,gc,argparse,hashlib
from torch.nn import functional as F
spec=importlib.util.spec_from_file_location('base185',B/'train.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
RECIPES=c.plan['recipes']
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

def train(o,cache,X,y_train,recipe,seed,out,predict_indices):
 started=time.monotonic();T=np.array(o['train']);assert len(y_train)==len(T);prefix,tail=c.model(out.parts[-4] if False else CURRENT_MODEL);prefixhash=c.statehash(prefix);del prefix;tail.requires_grad_(True)
 for m in tail.modules():
  if isinstance(m,nn.modules.batchnorm._BatchNorm):m.requires_grad_(False)
 initial=c.statehash(tail);bn={n:v.detach().clone() for n,v in tail.state_dict().items() if 'running_' in n or 'num_batches_tracked' in n};fixed={n:v.detach().clone() for n,v in tail.named_parameters() if not v.requires_grad};tail.eval().to('mps');h=old.head(X.shape[1]);h.load_state_dict(o['head']);assert c.statehash(h)==o['final_head_sha256'];initialhead=c.statehash(h);h.to('mps');mu,scale=old.tensors(o,'mps');yy=torch.tensor((y_train-o['target_mean'])/o['target_std'],dtype=torch.float32,device='mps');look={int(i):j for j,i in enumerate(T)};hp=list(h.parameters());tp=[p for p in tail.parameters() if p.requires_grad];groups=[{'params':hp,'lr':recipe['head_lr']},{'params':tp,'lr':recipe['tail_lr']}]
 opt=torch.optim.AdamW(groups,weight_decay=0.) if recipe['optimizer']=='AdamW' else torch.optim.SGD(groups,momentum=.9,dampening=0.,nesterov=False,weight_decay=0.)
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda ep:.1+.9*(1+math.cos(math.pi*ep/12))/2)
 rng=np.random.default_rng(seed+1);orders=[];updates=[];curve=[]
 def metric(epoch):
  pp=old.predict(h,tail,cache,X,T,o,'mps');curve.append({'epoch':epoch,'train_RMSE':float(np.sqrt(np.mean((y_train-pp)**2)))})
 metric(0);torch.manual_seed(seed+1000)
 for epoch in range(12):
  h.train();tail.eval();order=rng.permutation(T);orders.append(order);losses=[];assert all(not m.training for m in tail.modules());assert h.training
  for j in range(0,len(T),16):
   inds=order[j:j+16];opt.zero_grad(set_to_none=True)
   for start in range(0,len(inds),8):
    rows=inds[start:start+8];a=tail(torch.tensor(np.array(cache[rows]),device='mps'));prediction=h((a-mu)/scale).flatten();target=yy[[look[int(i)] for i in rows]];loss=F.smooth_l1_loss(prediction,target);assert torch.isfinite(loss);(loss*len(rows)/len(inds)).backward();losses.append(float(loss.detach()))
   norms=[]
   for params in [hp,tp]:norms.append(float(torch.stack([p.grad.detach().norm()**2 for p in params if p.grad is not None]).sum().sqrt()))
   assert all(math.isfinite(n) and n>0 for n in norms);grad=float(nn.utils.clip_grad_norm_(hp+tp,1.));assert math.isfinite(grad);assert all(p.grad is None for p in tail.parameters() if not p.requires_grad)
   lrs=[g['lr'] for g in opt.param_groups]
   with torch.no_grad():
    for g in opt.param_groups:
     for param in g['params']:param.mul_(1-g['lr']*.01)
   opt.step();updates.append([epoch,j//16,len(inds),*lrs,grad,*norms])
  scheduler.step()
  if (epoch+1)%3==0:metric(epoch+1);curve[-1]['train_augmented_mean_loss']=float(np.mean(losses))
 assert len(updates)==12*math.ceil(len(T)/16);assert c.statehash(tail)!=initial
 for n,v in tail.state_dict().items():
  if n in bn:assert torch.equal(v.cpu(),bn[n]),n
 for n,v in tail.named_parameters():
  if n in fixed:assert torch.equal(v.detach().cpu(),fixed[n]),n
 pred=old.predict(h,tail,cache,X,predict_indices,o,'mps')
 obj={**o,'head':{n:v.detach().cpu().clone() for n,v in h.state_dict().items()},'tail':{n:v.detach().cpu().clone() for n,v in tail.state_dict().items()},'head_initial':initialhead,'prefix_initial':prefixhash,'tail_initial':initial,'tail_final':c.statehash(tail),'final_head_sha256':c.statehash(h),'recipe':recipe,'adaptation_seed':seed,'order_adaptation_sha256':hashlib.sha256(np.stack(orders).tobytes()).hexdigest(),'update_count':len(updates),'curve':curve,'BN_and_prefix_frozen':True}
 torch.save(obj,_repo_io(out/'model.pt'));o2=ck(out/'model.pt');h.load_state_dict(o2['head']);tail.load_state_dict(o2['tail']);again=old.predict(h,tail,cache,X,predict_indices,o2,'mps');np.testing.assert_array_equal(pred,again)
 c.arrays(out/'predictions.npz',indices=predict_indices,prediction=pred);c.arrays(out/'updates.npz',values=np.array(updates),columns=['epoch','batch','rows','head_lr','tail_lr','preclip_total_gradnorm','preclip_head_gradnorm','preclip_tail_gradnorm'])
 rec={'recipe':recipe,'seconds':time.monotonic()-started,'updates':len(updates),'train_n':len(T),'prediction_n':len(predict_indices),'initial_train_RMSE':curve[0]['train_RMSE'],'final_train_RMSE':curve[-1]['train_RMSE'],'curve':curve,'gradient_norm_min':min(x[5] for x in updates),'gradient_norm_max':max(x[5] for x in updates),'gradient_clip_fraction':sum(x[5]>1 for x in updates)/len(updates),'head_lr_first_last':[updates[0][3],updates[-1][3]],'tail_lr_first_last':[updates[0][4],updates[-1][4]],'same_model_reload_exact':True,'BN_eval_affine_stats_unchanged':True,'nontrainable_gradients_none':True,'head_initial':initialhead,'order_sha256':obj['order_adaptation_sha256'],'model_sha256':c.sha(out/'model.pt'),'updates_sha256':c.sha(out/'updates.npz'),'predictions_sha256':c.sha(out/'predictions.npz')};c.write(out/'report.json',rec);c.emit({'path':str(out.relative_to(R)),**{k:rec[k] for k in ['seconds','updates','initial_train_RMSE','final_train_RMSE','gradient_clip_fraction']}})
 del tail,h,opt,groups,hp,tp,obj,o2;gc.collect();torch.mps.empty_cache();return pred,rec

def selection(y,T,innerpred):
 mse=np.mean((innerpred-y[T][None])**2,axis=1);winner=int(np.argmin(mse));families={'AdamW':int(np.argmin(mse[:2])),'SGD':2+int(np.argmin(mse[2:]))};return winner,families,mse

def run(name):
 global CURRENT_MODEL;CURRENT_MODEL=name;c.check();d=c.data('original');X=c.load(B/name/'features.npz')['X'].astype(float);cache=np.load(_repo_io(B/name/'prefix.npy'),mmap_mode='r');root=R/name;root.mkdir(exist_ok=True);
 if not (root/'decay-unit.json').exists():c.write(root/'decay-unit.json',decay_unit())
 start=time.monotonic();OOF={n:np.full(400,np.nan) for n in ['selected','AdamW','SGD']};selections=[]
 with c.Monitor(name+'-continued-run',3600) as mon:
  for k in range(5):
   fold=c.plan['folds'][str(k)];T=np.array(fold['train']);V=np.array(fold['validation']);path=root/f'fold-{k}';path.mkdir(exist_ok=True);ip=np.full((4,len(T)),np.nan);pos={int(i):j for j,i in enumerate(T)}
   for j,cell in enumerate(fold['inner']):
    I=np.array(cell['train']);H=np.array(cell['validation']);sub=path/f'inner-{j}';sub.mkdir(exist_ok=True);o=save_head(sub/'initial-head.pt',X,d['y'],I,H,202710185+k)
    for ri,recipe in enumerate(RECIPES):
     out=sub/recipe['name']
     if (out/'report.json').exists():
      rec=c.read(out/'report.json');assert c.sha(out/'model.pt')==rec['model_sha256'];saved=c.load(out/'predictions.npz');np.testing.assert_array_equal(saved['indices'],H);pp=saved['prediction'];c.emit({'reused':str(out.relative_to(R))})
     else:out.mkdir();pp,rec=train(o,cache,X,d['y'][I].copy(),recipe,202710186+100*k+j,out,H)
     ip[ri,[pos[int(i)] for i in H]]=pp;mon.sample()
     if k==j==ri==0:
      projected=rec['seconds']*(60+10*320/len(I))+120;gate=projected<3600;c.write(root/'first-fit-benchmark-revised.json',{'passed':gate,'projected_model_seconds_including_head_allowance':projected,'first_fit_seconds':rec['seconds'],'microbatch':8,'effective_batch':16,'resources':mon.samples[-1]});assert gate,'firstfitcomputeprojectionexceedsbudget'
   assert np.isfinite(ip).all();winner,families,mse=selection(d['y'],T,ip);poison=d['y'].copy();poison[V]=poison[V]*-1000+1e6;w2,f2,m2=selection(poison,T,ip);assert winner==w2 and families==f2;np.testing.assert_array_equal(mse,m2);c.arrays(path/'inner-OOF.npz',train=T,predictions=ip,MSE=mse);rec={'fold':k,'winner_index':winner,'winner':RECIPES[winner],'family_winners':families,'inner_RMSE':np.sqrt(mse).tolist(),'outer_label_poison_selection_exact':True};c.write(path/'selection.json',rec);selections.append(rec);c.emit({'selected':name,**rec})
   o=ck(B/name/'original'/f'fold-{k}/frozen100.pt');assert o['train']==T.tolist() and o['validation']==V.tolist()
   for fam,ri in families.items():
    out=path/('outer-'+fam);out.mkdir();pp,_=train(o,cache,X,d['y'][T].copy(),RECIPES[ri],202710186+100*k+99,out,V);OOF[fam][V]=pp
    if winner==ri:OOF['selected'][V]=pp
  assert all(np.isfinite(pp).all() for pp in OOF.values());c.arrays(root/'OOF.npz',ids=d['ids'],y=d['y'],groups=d['g'],fold=d['fold'],best021=d['blend_L_MLP'],**OOF);report={'model':name,'scores':{n:old.evaluate(d,pp) for n,pp in OOF.items()},'selections':selections,'OOF_sha256':c.sha(root/'OOF.npz'),'primary':'selected; optimizerfamilyOOFdiagnosticonly; neverchoosefamilyusingouterlabels','new_inner_fits':60,'new_outer_fits':10,'seconds':time.monotonic()-start};c.write(root/'report.json',report);c.emit({'completed':name,'scores':report['scores'],'seconds':report['seconds']})
 c.check()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('model',choices=['resnet50','efficientnet_b0']);a=ap.parse_args();run(a.model)
