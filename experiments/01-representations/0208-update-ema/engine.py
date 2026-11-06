from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import np,torch,R,P,time
F=torch.nn.functional
BETA=.99

def pack(params,opt):
 values=[p.detach().flatten() for p in params]
 for key in ['exp_avg','exp_avg_sq']:values.extend(opt.state[p][key].detach().flatten() for p in params)
 flat=torch.cat(values).cpu().clone();steps=[int(opt.state[p]['step'].item()) for p in params];groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups]
 return flat,steps,groups

def cpu_tree(obj):
 if torch.is_tensor(obj):return obj.detach().cpu().clone()
 if isinstance(obj,dict):return {k:cpu_tree(v) for k,v in obj.items()}
 if isinstance(obj,list):return [cpu_tree(v) for v in obj]
 if isinstance(obj,tuple):return tuple(cpu_tree(v) for v in obj)
 return obj

def unpack(vector,named):
 out={};offset=0
 for name,p in named:
  out[name]=vector[offset:offset+p.numel()].reshape(p.shape).clone();offset+=p.numel()
 assert offset==vector.numel();return out

@torch.inference_mode()
def predict(m,images,ix,norm,dev):
 m.eval();out=[]
 for j in range(0,len(ix),32):
  x=(images[ix[j:j+32]]-norm['x_mean'])/max(norm['x_std'],1e-6);z=m(torch.from_numpy(np.ascontiguousarray(x)).to(dev)).cpu().numpy();out.extend((z*norm['y_std']+norm['y_mean']).tolist())
 return np.array(out)

def fit(d,k):
 folder=R/f'fold-{k}';folder.mkdir(exist_ok=False);start=time.perf_counter();T,V=c.indices(k);norm=c.norm(d,T);oldfolder=P/f'experiments/contrast-augmentation-206/fold-{k}-control';oldfit=c.read(oldfolder/'fit.json');oldstreams=c.load(oldfolder/'streams.npz');oldmodel=torch.load(_repo_io(oldfolder/'model.pt'),map_location='cpu',weights_only=True)
 assert oldmodel['seed']==20261003+k and oldmodel['normalization']==norm and oldmodel['train_ids']==d['ids'][T].tolist() and oldmodel['validation_ids']==d['ids'][V].tolist();X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];torch.manual_seed(20261003+k);m=c.legacy.TinyCNN().to(c.device);named=list(m.named_parameters());params=[p for _,p in named];N=sum(p.numel() for p in params);assert N==91905
 bn=sum(isinstance(mod,torch.nn.modules.batchnorm._BatchNorm) for mod in m.modules());gn=sum(isinstance(mod,torch.nn.GroupNorm) for mod in m.modules());assert bn==0 and gn==4 and list(m.named_buffers())==[];assert set(m.state_dict())==set(n for n,_ in named);initial=c.statehash(m.state_dict());assert initial==oldfit['initial_state_SHA256'];rngstart=c.rnghash();assert rngstart==oldfit['rng_start']
 # This independent parameter list has no forward model or backward pass.
 shadow=torch.nn.ParameterList([torch.nn.Parameter(p.detach().clone()) for p in params]);shadowparams=list(shadow.parameters());o=torch.optim.AdamW(params,lr=.002,weight_decay=.01);so=torch.optim.AdamW(shadowparams,lr=.002,weight_decay=.01);sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=80,eta_min=.0001);ssch=torch.optim.lr_scheduler.CosineAnnealingLR(so,T_max=80,eta_min=.0001);assert not o.state and not so.state
 ema=torch.cat([p.detach().flatten() for p in params]).cpu().clone();theta0=ema.clone();direct=theta0.double()*(BETA**800);torch.save(unpack(ema,named),_repo_io(folder/'ema-update-0.pt'));rng=np.random.default_rng(20261003+k);avg=None;orderlog=[];rotlog=[];fliplog=[];lrlog=[];losslog=[];rnglog=[];ih={};eph={};proof=[];counter=0
 c.emit({'fit_start':k,'EMA_decay':BETA,'BatchNorm_count':bn,'buffers':0})
 for epoch in range(80):
  order=rng.permutation(320)
  for first in range(0,320,32):
   ix=order[first:first+32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);xb=np.rot90(X[ix].copy(),rot,axes=(2,3));xb=xb[:,:,:,::-1] if flip else xb;xb=np.ascontiguousarray(xb);n=counter;state=c.rnghash()
   np.testing.assert_array_equal(ix,oldstreams['order'][n]);assert rot==oldstreams['rot'][n] and flip==oldstreams['flip'][n] and o.param_groups[0]['lr']==oldstreams['lr'][n];assert state['CPU']==oldstreams['rng_CPU'][n] and state.get('MPS','CPU only')==oldstreams['rng_MPS'][n]
   if n in [0,99,399,799]:ih[str(n)]=c.ah(xb);assert ih[str(n)]==oldfit['input_hashes'][str(n)]
   orderlog.append(ix.copy());rotlog.append(rot);fliplog.append(flip);lrlog.append(o.param_groups[0]['lr']);rnglog.append(state);m.train();o.zero_grad(set_to_none=True);so.zero_grad(set_to_none=True);loss=F.mse_loss(m(torch.from_numpy(xb).to(c.device)),torch.tensor(y[ix],dtype=torch.float32,device=c.device));assert torch.isfinite(loss);loss.backward();gradnorm=torch.nn.utils.clip_grad_norm_(params,5.);assert torch.isfinite(gradnorm)
   for p,sp in zip(params,shadowparams):sp.grad=p.grad.detach().clone()
   opt_rng=c.rnghash();o.step();so.step();assert c.rnghash()==opt_rng;counter+=1
   # Compare the live optimizer with the no-EMA optimizer replay after every update.
   live,steps,groups=pack(params,o);reference,rsteps,rgroups=pack(shadowparams,so);assert torch.equal(live,reference) and steps==rsteps==[counter]*len(params) and groups==rgroups
   before_rng=c.rnghash();theta=live[:N].clone();ema.mul_(BETA).add_(theta,alpha=1-BETA);direct.add_(theta.double(),alpha=(1-BETA)*BETA**(800-counter));assert torch.isfinite(ema).all()
   after,aftersteps,aftergroups=pack(params,o);assert torch.equal(live,after) and steps==aftersteps and groups==aftergroups and c.rnghash()==before_rng
   values=[float(loss.detach().cpu()),float(gradnorm.detach().cpu())];np.testing.assert_array_equal(values,oldstreams['values'][n]);losslog.append(values)
   if counter in [1,10,100,400,800]:
    torch.save(unpack(ema,named),_repo_io(folder/f'ema-update-{counter}.pt'));proof.append({'update':counter,'live_optimizer_parameters_moments_SHA256':c.ah(live.numpy()),'reference_SHA256':c.ah(reference.numpy()),'optimizer_step_counters':steps,'groups':groups,'EMA_state_SHA256':c.ah(ema.numpy()),'post_update_theta_SHA256':c.ah(theta.numpy()),'optimizer_reference_exact':True,'EMA_side_effects_absent':True})
  sch.step();ssch.step();assert sch.state_dict()==ssch.state_dict()
  if epoch>=75:
   sd={n:v.detach().cpu().clone() for n,v in m.state_dict().items()};eph[str(epoch+1)]=c.statehash(sd);assert eph[str(epoch+1)]==oldfit['epoch_state_SHA256'][str(epoch+1)];torch.save(sd,_repo_io(folder/f'epoch-{epoch+1}.pt'))
   if avg is None:avg={n:v/5 for n,v in sd.items()}
   else:
    for n,v in sd.items():avg[n]+=v/5
  if (epoch+1)%20==0:c.emit({'fold':k,'epoch':epoch+1,'seconds':time.perf_counter()-start,'optimizer_reference_checks':counter,'EMA_side_effect_checks':counter})
  assert time.perf_counter()-start<240
 assert counter==800 and c.rnghash()==oldfit['rng_end'];assert all(torch.equal(v,oldmodel['state_dict'][n]) for n,v in avg.items());directerr=float(abs(ema.double()-direct).max());assert directerr<=1e-5;emasd=unpack(ema,named);directsd=unpack(direct.float(),named)
 torch.save({'optimizer':cpu_tree(o.state_dict()),'reference_optimizer':cpu_tree(so.state_dict()),'scheduler':sch.state_dict(),'reference_scheduler':ssch.state_dict(),'proof_samples':proof},_repo_io(folder/'optimizer-verification.pt'));torch.save(directsd,_repo_io(folder/'EMA-direct-float64-rounded.pt'))
 predictions={}
 for arm,sd in [('last5_uniform',avg),('update_ema',emasd)]:
  m.load_state_dict(sd);p=predict(m,d['images'],V,norm,c.device);assert np.isfinite(p).all();predictions[arm]=p;torch.save({'state_dict':sd,'normalization':norm,'train_ids':d['ids'][T].tolist(),'validation_ids':d['ids'][V].tolist(),'train_indices':T.tolist(),'validation_indices':V.tolist(),'fold':k,'seed':20261003+k,'arm':arm,'updates':800},_repo_io(folder/f'{arm}.pt'))
 ref=c.load(P/'experiments/contrast-augmentation-206/frozen-reference.npz');np.testing.assert_array_equal(predictions['last5_uniform'],ref['old_first_CNN'][V]);c.arrays(folder/'predictions.npz',validation_indices=V,**predictions)
 c.arrays(folder/'streams.npz',order=np.stack(orderlog),rot=np.array(rotlog),flip=np.array(fliplog),lr=np.array(lrlog),values=np.array(losslog),rng_CPU=np.array([r['CPU'] for r in rnglog]),rng_MPS=np.array([r.get('MPS','CPU only') for r in rnglog]),EMA_update_indices=np.arange(1,801))
 rec={'fold':k,'seed':20261003+k,'gradient_updates':800,'optimizer_verification_steps':800,'EMA_updates':800,'normalization':norm,'initial_state_exact206':True,'sample_D4_LR_RNG_loss_gradient_norm_exact_all800':True,'last5_epoch_states_and_average_exact206':True,'last5_uniform_OOF_exact':True,'optimizer_reference_exact_checks':800,'EMA_no_side_effect_checks':800,'proof_samples':proof,'BatchNorm_modules':bn,'GroupNorm_modules':gn,'buffers':[],'EMA_direct_state_max_error':directerr,'seconds':time.perf_counter()-start,'performance_scores_computed':False,'source206_control_fit_SHA256':c.sha(oldfolder/'fit.json'),'model_SHA256':{a:c.sha(folder/f'{a}.pt') for a in predictions},'streams_SHA256':c.sha(folder/'streams.npz'),'predictions_SHA256':c.sha(folder/'predictions.npz')};c.write(folder/'fit.json',rec);c.emit({'fold_complete':k,'seconds':rec['seconds'],'EMA_direct_error':directerr});del m,o,so,shadow,avg,ema,direct;__import__('gc').collect()
 if c.device=='mps':torch.mps.empty_cache()
 return predictions,rec
