"""Predeclared local MPS fallback for both matched dense sampling arms."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import cnn_repair as cr
import sys,time,json,gc
import numpy as np,torch
from torch import nn
from threadpoolctl import threadpool_limits

@torch.no_grad()
def infer(model,im,ctx,ids,ys):
 model.eval();out=[]
 for a in range(0,len(ids),32):
  ix=torch.as_tensor(ids[a:a+32],device='mps');x=im[ix];cc=ctx[ix];views=[]
  for rot,flip in [(0,False),(1,False),(0,True),(1,True)]:
   xx=torch.rot90(x,rot,[2,3]);xx=torch.flip(xx,[3]) if flip else xx;p,_=model(xx,cc,'aux');views.append(p.detach().cpu().numpy())
  out.extend(np.mean(views,axis=0))
 return np.array(out,dtype=float)*ys

def fit(z,kind,seed,path,epochs=40):
 start=time.monotonic();torch.manual_seed(seed);model=cr.DenseFirst(kind).to('mps');torch.mps.manual_seed(seed);rng=np.random.default_rng(seed);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=.0001)
 im=torch.from_numpy(z['images']).to('mps');ctx=torch.from_numpy(z['context']).to('mps');target=torch.from_numpy(z['targets']['residual']).to('mps');direct=torch.from_numpy(z['targets']['direct']).to('mps');tr=z['train'];ys=z['meta']['target_std'];history=[];avg=None;peak=0
 for ep in range(epochs):
  model.train();order=rng.permutation(len(tr));se=0.;sa=0.
  for a in range(0,len(tr),32):
   j=order[a:a+32];ix=torch.as_tensor(tr[j],device='mps');jj=torch.as_tensor(j,device='mps');rot=int(rng.integers(4));flip=bool(rng.random()<.5);x=torch.rot90(im[ix],rot,[2,3]);x=torch.flip(x,[3]) if flip else x;pred,aux=model(x,ctx[ix],'aux');main=nn.functional.mse_loss(pred,target[jj]);secondary=nn.functional.mse_loss(aux,direct[jj]);loss=main+.2*secondary;assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);opt.step();se+=float(main.detach().cpu())*len(j);sa+=float(secondary.detach().cpu())*len(j)
  sched.step();mem=c.runtime();peak=max(peak,mem['rss_bytes']+mem['mps_driver_bytes']);history.append(dict(epoch=ep+1,augmented_train_rmse=float(np.sqrt(se/len(tr))*ys),auxiliary_rmse=float(np.sqrt(sa/len(tr))*ys)))
  if ep>=epochs-5:
   state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   if avg is None:avg={k:v/5 for k,v in state.items()}
   else:
    for k,v in state.items():avg[k]+=v/5
  if (ep+1)%20==0:print(kind,seed,ep+1,round(time.monotonic()-start,1),flush=True)
 model.load_state_dict(avg);p=infer(model,im,ctx,z['valid'],ys);path.mkdir(parents=True,exist_ok=False);torch.save(dict(state_dict=avg,normalization=z['meta'],kind=kind,seed=seed,epochs=epochs,device='mps',parameter_count=sum(p.numel() for p in model.parameters())),_repo_io(path/'model.pt'));new=cr.DenseFirst(kind).to('mps');new.load_state_dict(torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True)['state_dict']);replay=infer(new,im,ctx,z['valid'],ys);err=float(abs(replay-p).max());assert err<1e-6,err;c.arrays(str((path/'predictions.npz').relative_to(c.HERE)),validation_index=z['valid'],train_index=tr,validation_prediction=p);torch.mps.synchronize();elapsed=time.monotonic()-start;c.save(_repo_io(str((path/'training.json').relative_to(c.HERE))),dict(kind=kind,seed=seed,epochs=epochs,meta=z['meta'],history=history,seconds=elapsed,reload_maxdiff=err,max_combined_memory=peak,parameter_count=sum(p.numel() for p in model.parameters())));del model,new,opt,sched,im,ctx,target,direct,avg;gc.collect();torch.mps.empty_cache();return elapsed

def probe():
 assert not json.loads((c.HERE/'dense-resource-probe.json').read_text())['passes'];d,y=cr.load();z=cr.data(d,y,0,'selector','073');z['valid']=z['valid'][:1];start=time.monotonic();seconds=fit(z,'pool',2026101501,c.HERE/'dense-mps-resource-probe',5);projection=seconds*8*40*1.25
 c.save(_repo_io('dense-mps-resource-probe.json'),dict(seconds=time.monotonic()-start,fit_seconds=seconds,projected_main_seconds=projection,main_fits=40,device='mps',passes=projection<3000,query_labels_used=False,code_sha256=c.sha(__file__)));
 print('MPS DENSE PROJECTION',projection,flush=True)

def train():
 assert json.loads((c.HERE/'dense-mps-resource-probe.json').read_text())['passes'];d,y=cr.load();start=time.monotonic();c.save(_repo_io('dense-code-lock.json'),dict(code=c.sha(__file__),constructor=c.sha(c.HERE/'cnn_repair.py'),plan=c.sha(c.HERE/'PLAN.md'),device='mps',dtype='float32',reason='CPU projection6336s exceeds the fixed budget; predeclared resource fallback for BOTH arms',arms=['pick','pool'],seeds=[2026101501,2026201501]));records=[]
 for k in range(5):
  for stage in ['selector','final']:
   z=cr.data(d,y,k,stage,'073');outside=np.setdiff1d(np.arange(500),z['train']);alt=y.copy();alt[outside]=987654.;check=cr.data(d,alt,k,stage,'073');assert z['meta']==check['meta'];assert all(np.array_equal(z['targets'][a],check['targets'][a]) for a in z['targets']);assert np.array_equal(z['context'],check['context']);del check
   for seed in [0,1]:
    for kind in ['pick','pool']:
     assert time.monotonic()-start<3300,'fixed55min cap';p=c.HERE/f'cnn-dense-{kind}/seed{seed}/fold{k}-{stage}';sec=fit(z,kind,2026101501+100*k+100000*seed,p);records.append(dict(fold=k,stage=stage,seed=seed,kind=kind,seconds=sec));print('DENSE DONE',k,stage,seed,kind,round(time.monotonic()-start,1),flush=True)
   del z;gc.collect()
 c.save(_repo_io('dense-complete.json'),dict(complete=True,records=records,seconds=time.monotonic()-start,device='mps',source_sha256=c.sha(__file__)))

if __name__=='__main__':
 assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):{'probe':probe,'train':train}[sys.argv[1]]()
