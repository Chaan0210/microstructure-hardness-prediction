from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,np,torch
c.check();d=c.data();ref=c.load(P/'experiments/contrast-augmentation-206/frozen-reference.npz');pred={n:np.full(400,np.nan) for n in ['uniform','tail_ema']};checks=[]
@torch.inference_mode()
def predict(m,ix,norm,dev):
 m.eval();out=[]
 for first in range(0,len(ix),32):
  x=(d['images'][ix[first:first+32]]-norm['x_mean'])/max(norm['x_std'],1e-6);z=m(torch.from_numpy(np.ascontiguousarray(x)).to(dev)).cpu().numpy();out.extend((z*norm['y_std']+norm['y_mean']).tolist())
 return np.array(out)
with c.Monitor('inference',300):
 assert c.read(P/'experiments/contrast-augmentation-206/validation.json')['passed'];np.testing.assert_array_equal(d['ids'],ref['ids']);np.testing.assert_array_equal(d['blend_L_MLP'],ref['best021'])
 for k in range(5):
  T,V=c.indices(k);folder=P/f'experiments/contrast-augmentation-206/fold-{k}-control';old=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);norm=c.norm(d,T);assert old['normalization']==norm;assert old['train_ids']==d['ids'][T].tolist() and old['validation_ids']==d['ids'][V].tolist();assert not set(d['g'][T])&set(d['g'][V]);fit=c.read(folder/'fit.json');assert fit['model_SHA256']==c.sha(folder/'model.pt');assert c.read(folder/'control-reproduction.json')['passed'];uniform=None;ema=None;states=[]
  for ep in range(76,81):
   path=folder/f'epoch-{ep}.pt';state=torch.load(_repo_io(path),map_location='cpu',weights_only=True);assert c.statehash(state)==fit['epoch_state_SHA256'][str(ep)];assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values());states.append(state)
   if uniform is None:uniform={n:v/5 for n,v in state.items()};ema={n:torch.zeros_like(v) for n,v in state.items()}
   else:
    for n,v in state.items():uniform[n]+=v/5
   for n,v in state.items():ema[n]=.5*ema[n]+.5*v
  for n in ema:ema[n]=ema[n]/(1-.5**5)
  legacy=torch.load(_repo_io(P/f'artifacts/run-009/fold-{k}.pt'),map_location='cpu',weights_only=True)
  assert all(torch.equal(v,old['state_dict'][n]) and torch.equal(v,legacy['state_dict'][n]) for n,v in uniform.items())
  direct={};maxerr=0.;inside=True
  for n in ema:
   values=np.stack([s[n].numpy().astype(np.float64) for s in states]);w=np.array([1,2,4,8,16],np.float64)/31;out=np.tensordot(w,values,axes=(0,0));direct[n]=torch.from_numpy(np.asarray(out,dtype=np.float32));maxerr=max(maxerr,float(abs(ema[n].double()-torch.from_numpy(np.asarray(out))).max()));inside=inside and bool(np.all(ema[n].numpy()>=values.min(0)-1e-6) and np.all(ema[n].numpy()<=values.max(0)+1e-6))
  assert maxerr<=1e-6 and inside
  # Held-out targets cannot affect checkpoint averaging; train-only normalization remains identical.
  poisoned=dict(d);poisoned['y']=d['y'].copy();poisoned['y'][V]=np.nan;assert c.norm(poisoned,T)==norm
  rec={'fold':k,'uniform_state_exact':True,'epoch_state_hashes_verified':True,'train_only_normalization_exact':True,'poisoned_validation_y_normalization_exact':True,'EMA_float64_direct_state_error':maxerr,'EMA_within_componentwise_convex_hull':inside,'reload':{}}
  output=R/f'fold-{k}';output.mkdir(exist_ok=False)
  for arm,sd in [('uniform',uniform),('tail_ema',ema)]:
   m=c.legacy.TinyCNN();assert sum(p.numel() for p in m.parameters())==91905;assert list(m.named_buffers())==[];assert set(sd)==set(dict(m.named_parameters()));m.load_state_dict(sd,strict=True);m.requires_grad_(False);m.eval();h=c.statehash(m.state_dict());m=m.to(c.device);p=predict(m,V,norm,c.device);assert np.isfinite(p).all() and c.statehash(m.state_dict())==h;pred[arm][V]=p
   torch.save({'state_dict':sd,'normalization':norm,'train_ids':d['ids'][T].tolist(),'validation_ids':d['ids'][V].tolist(),'seed':20261003+k,'arm':arm,'epochs':[76,77,78,79,80]},_repo_io(output/f'{arm}.pt'));saved=torch.load(_repo_io(output/f'{arm}.pt'),map_location='cpu',weights_only=True);m.load_state_dict(saved['state_dict']);again=predict(m,V,norm,c.device);np.testing.assert_array_equal(p,again);m=m.to('cpu');cpu=predict(m,V,norm,'cpu');cpuerr=float(abs(cpu-p).max());assert cpuerr<=.001 and c.statehash(m.state_dict())==h
   if arm=='uniform':np.testing.assert_array_equal(p,ref['old_first_CNN'][V])
   if arm=='tail_ema':m.load_state_dict(direct);ip=predict(m,V,norm,'cpu');inderr=float(abs(ip-cpu).max());assert inderr<=.001;rec['EMA_independent_state_prediction_error']=inderr
   rec['reload'][arm]={'MPS_repeat_max_error':float(abs(again-p).max()),'CPU_MPS_max_error':cpuerr,'state_unchanged':True,'state_SHA256':h,'model_SHA256':c.sha(output/f'{arm}.pt')};del m
  rec['all_buffers_empty_GroupNorm_only']=True;c.write(output/'validation.json',rec);checks.append(rec);c.emit({'fold':k,'uniform_reproduced_exact':True,'direct_EMA_state_error':maxerr,'CPU_errors':rec['reload']})
  if c.device=='mps':torch.mps.empty_cache()
 assert all(np.isfinite(p).all() for p in pred.values());qa=ref['best021']+ref['coefficient']*(pred['uniform']-ref['old_first_CNN']);qb=ref['best021']+ref['coefficient']*(pred['tail_ema']-ref['old_first_CNN']);np.testing.assert_array_equal(qa,ref['best021'])
 for k in range(5):
  T,V=c.indices(k);np.testing.assert_array_equal(ref['coefficient'][V],np.full(80,c.plan['fixed_ensemble']['coef_by_fold'][str(k)]))
 c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],**pred,fixed_uniform=qa,fixed_tail_ema=qb);c.write(R/'validation.json',{'passed':True,'folds':checks,'OOF_SHA256':c.sha(R/'OOF.npz'),'fixed_uniform_exact_best021':True,'new_gradient_fits':0,'optimizer_steps':0,'individual_epoch_predictions_or_scores_computed':False,'performance_scores_computed':False,'torch':torch.__version__,'numpy':np.__version__,'device':c.device,'plan_SHA256':c.sha(R/'plan.json')})
c.check()
