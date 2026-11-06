"""Supervised encoder adaptation with an analytic ridge leave-one-out training loss.

The leave-one-out quantity is a TRAINING objective, not held-out validation of
the learned encoder. The outer query images/labels never enter optimization.
Gradient caching keeps the whole-image training objective within local memory.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['TMPDIR']=str(_repo_path(HERE, 'tmp'))
import argparse,hashlib,json,math,resource,time
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
import micronet_direct as direct

class Encoder(nn.Module):
    def __init__(self,arm):
        super().__init__();self.body,self.provenance=direct.backbone(arm)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def forward(self,gray):
        f=self.body((gray.expand(-1,3,-1,-1)-self.mean)/self.std)
        raw=torch.stack([gray.pow(k).mean((1,2,3)) for k in [1,2,3,4]],1);return torch.cat([f,raw],1)

def batch(px,ix,rotations,device):
    x=np.stack([np.ascontiguousarray(np.rot90(px[j,0],int(r))) for j,r in zip(ix,rotations)])[:,None]
    return torch.tensor(x.astype(np.float32)/255.,device=device)

@torch.no_grad()
def extract(model,px,ix,device,batch_size,rotations=None):
    model.eval();out=[]
    for first in range(0,len(ix),batch_size):
        jj=ix[first:first+batch_size]
        if rotations is None:
            values=[model(batch(px,jj,np.full(len(jj),r),device)).cpu().numpy() for r in [0,1,2,3]];f=np.mean(values,0)
        else:f=model(batch(px,jj,rotations[first:first+batch_size],device)).cpu().numpy()
        out.append(f)
    return np.concatenate(out)

def normalize(X):
    mu=X.mean(0);sd=np.maximum.reduce([X.std(0),.1*np.sqrt(np.mean(X*X,0)),np.full(X.shape[1],1e-4)])
    return np.clip((X-mu)/sd,-8,8),mu,sd

def fit_head(F,y,alpha):
    X,mu,sd=normalize(F.astype(float));n=len(X);K=X@X.T/X.shape[1]
    A=np.linalg.solve(K+alpha*np.eye(n),np.eye(n));v=A.sum(1);den=v.sum();C=A-np.outer(v,v)/den
    dual=C@y;w=X.T@dual/X.shape[1];bias=float(v@y/den)
    return dict(mean=mu,scale=sd,weight=w,bias=bias,alpha=alpha)

def predict_head(head,F):return np.clip((F-head['mean'])/head['scale'],-8,8)@head['weight']+head['bias']

def select_alpha(F,y,seed):
    alphas=[.0001,.001,.01,.1,1.,10.];sse=np.zeros(len(alphas))
    for tr,q in KFold(3,shuffle=True,random_state=seed).split(F):
        for j,a in enumerate(alphas):sse[j]+=np.sum((predict_head(fit_head(F[tr],y[tr],a),F[q])-y[q])**2)
    return alphas[int(np.argmin(sse))],dict(alphas=alphas,inner_rmse=np.sqrt(sse/len(y)).tolist())

def training_proxy(F,y,alpha):
    mu=F.mean(0);sd=(F.var(0,correction=0)+1e-12).sqrt();rms=(F.square().mean(0)+1e-12).sqrt()
    scale=torch.maximum(torch.maximum(sd,.1*rms),torch.full_like(sd,1e-4));X=((F-mu)/scale).clamp(-8,8)
    K=X@X.T/X.shape[1];A=torch.cholesky_inverse(torch.linalg.cholesky(K+alpha*torch.eye(len(F),dtype=F.dtype)))
    v=A.sum(1);C=A-v[:,None]*v[None,:]/v.sum();residual=(C@y)/C.diagonal().clamp(min=1e-9)
    return residual.square().mean()

def check_gradient():
    # Compare two-pass gradient caching with direct differentiation on actual train pixels.
    torch.manual_seed(7);d=np.load(_repo_path(HERE, 'data.npz'));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');ix=d['dev'][d['fold']!=0][:4]
    a=Encoder('micronet').eval();b=Encoder('micronet').eval();b.load_state_dict(a.state_dict());x=batch(px,ix,np.zeros(4,int),'cpu');y=torch.tensor((d['y'][ix]-d['y'][ix].mean())/d['y'][ix].std(),dtype=torch.float64)
    f=a(x);loss=training_proxy(f.double(),y,.1);loss.backward()
    with torch.no_grad():cache=b(x)
    z=cache.double().requires_grad_(True);other=training_proxy(z,y,.1);other.backward();b(x).backward(z.grad.float())
    errors=[float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),b.parameters()) if p.grad is not None]
    error=max(errors);assert error<1e-5
    r=dict(loss_difference=abs(float(loss.detach()-other.detach())),max_parameter_gradient_difference=error,original_training_ids=d['ids'][ix].tolist(),optimizer_updates=0)
    direct.save(_repo_path(HERE, 'deep-kernel-gradient-check.json'),r);print(r,flush=True)

def run(a):
    device=a.device;d=np.load(_repo_path(HERE, 'data.npz'));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');tr=d['dev'][d['fold']!=a.fold];q=d['dev'][d['fold']==a.fold]
    torch.manual_seed(a.seed)
    if device=='mps':torch.mps.manual_seed(a.seed);torch.mps.set_per_process_memory_fraction(.35)
    model=Encoder(a.arm).to(device).eval();prior=[p.detach().clone() for p in model.body.parameters()]
    out=_repo_path(HERE, 'runs', a.tag);out.mkdir(parents=True,exist_ok=True);assert not (out/'done.json').exists()
    cache=_repo_path(HERE, f'microscopy-contrast/{a.arm}.npz')
    if cache.exists():
        c=np.load(_repo_io(cache));assert np.array_equal(c['ids'],d['ids']);photo=c['photo']
        allf=np.c_[c['raw'],photo[:,0],photo[:,1]**2+photo[:,0]**2,photo[:,2:4]].astype(np.float32)
        fresh=extract(model,px,tr[:3],device,a.batch);cache_error=float(abs(fresh-allf[tr[:3]]).max());assert cache_error<1e-4
        F0=allf[tr];Q0=allf[q];print('INITIAL_CACHE_MAX_ERROR',cache_error,flush=True)
    else:F0=extract(model,px,tr,device,a.batch);Q0=extract(model,px,q,device,a.batch)
    alpha,choice=select_alpha(F0,d['y'][tr],2026101913+a.fold)
    head=fit_head(F0,d['y'][tr],alpha);initial=predict_head(head,Q0);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());y=torch.tensor((d['y'][tr]-ym)/ys,dtype=torch.float64)
    recipe=vars(a)|dict(train=tr.tolist(),query=q.tolist(),train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist(),alpha=alpha,alpha_selection=choice,
        pretrained=model.provenance,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        objective='Training-only analyticridgeLOOpredictionloss;notanunbiasedvalidationscoreoflearnedfeatures. Outerqueryimagesandlabelsneverenterfitting.',
        preprocessing='Currenttraining-featuremean/RMSfloorstandardization,clip±8;C4sampledtrainingviews,C4meaninference;allBNrunningstatsfrozen.',
        optimizer='Onefull-original-imagegradientstepusingtwo-passfeaturegradientcaching;L2anchoronencoderpretrainedparameters.')
    direct.save(_repo_io(out/'recipe.json'),recipe);np.savez(_repo_io(out/'initial.npz'),train=tr,query=q,prediction=initial);print('INITIAL',direct.rmse(initial,d['y'][q]),'alpha',alpha,flush=True)
    opt=torch.optim.AdamW(model.body.parameters(),lr=a.lr,weight_decay=0.);rng=np.random.default_rng(a.seed);start=time.monotonic();curve=[];first=0;elapsed=0;average=None;navg=0
    if (out/'resume.pt').exists():
        r=torch.load(_repo_io(out/'resume.pt'),map_location='cpu',weights_only=False);assert r['recipe']==recipe;model.load_state_dict(r['state']);opt.load_state_dict(r['optimizer']);rng.bit_generator.state=r['rng'];first=r['step'];curve=r['curve'];elapsed=r['elapsed'];average=r['average'];navg=r['navg']
        for state in opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value) and key!='step':state[key]=value.to(device)
    for step in range(first,a.steps):
        model.eval();rotations=rng.integers(4,size=len(tr));F=extract(model,px,tr,device,a.batch,rotations)
        feature=torch.tensor(F,dtype=torch.float64,requires_grad=True);loss=training_proxy(feature,y,alpha);assert torch.isfinite(loss);loss.backward();grad=feature.grad.float().numpy()
        opt.zero_grad(set_to_none=True)
        for j in range(0,len(tr),a.batch):
            z=model(batch(px,tr[j:j+a.batch],rotations[j:j+a.batch],device));z.backward(torch.tensor(grad[j:j+a.batch],device=device))
        anchor=a.anchor*sum((p-old).square().sum() for p,old in zip(model.body.parameters(),prior));anchor.backward();nn.utils.clip_grad_norm_(model.parameters(),5.)
        for group in opt.param_groups:group['lr']=a.lr*(.05+.95*.5*(1+math.cos(math.pi*step/a.steps)))
        opt.step()
        if device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
        if step>=a.steps-10:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};navg+=1
            if average is None:average=state
            else:
                for k,v in state.items():
                    if v.is_floating_point():average[k].add_((v-average[k])/navg)
                    else:average[k]=v
        rec=dict(step=step+1,training_LOO_proxy_rmse_before_update=float(loss.detach().sqrt())*ys,seconds=elapsed+time.monotonic()-start)
        if step==0 or (step+1)%10==0 or step+1==a.steps:
            T=extract(model,px,tr,device,a.batch);Q=extract(model,px,q,device,a.batch);h=fit_head(T,d['y'][tr],alpha);pred=predict_head(h,Q)
            rec.update(validation_rmse=direct.rmse(pred,d['y'][q]),training_fitted_rmse=direct.rmse(predict_head(h,T),d['y'][tr]));np.savez(_repo_io(out/f'step{step+1:03d}.npz'),query=q,prediction=pred)
            record=dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),recipe=recipe,step=step+1,curve=curve+[rec],rng=rng.bit_generator.state,elapsed=rec['seconds'],average=average,navg=navg)
            torch.save(record,_repo_io(out/'resume-new.pt'));(out/'resume-new.pt').replace(out/'resume.pt');print(json.dumps(rec),flush=True)
        curve.append(rec);direct.save(_repo_io(out/'curve.json'),curve)
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+(torch.mps.driver_allocated_memory() if device=='mps' else 0);assert mem<6*2**30
    model.load_state_dict(average);T=extract(model,px,tr,device,a.batch);Q=extract(model,px,q,device,a.batch);head=fit_head(T,d['y'][tr],alpha);pred=predict_head(head,Q)
    torch.save(dict(state=average,recipe=recipe),_repo_io(out/'model.pt'));np.savez(_repo_io(out/'head.npz'),**head);np.savez(_repo_io(out/'predictions.npz'),train=tr,query=q,prediction=pred,train_features=T,query_features=Q)
    ref=np.load(_repo_path(ROOT, 'experiments/rmse-research-20261015/inputs.npz'));ix=d['fold']==a.fold;base=d['base083'][ix];p21=ref['C73_cnn'][0,ix];eligible=~d['failed'][ix]
    r=dict(rmse=direct.rmse(pred,d['y'][q]),frozen_rmse=direct.rmse(initial,d['y'][q]),G021_rmse=direct.rmse(p21,d['y'][q]),base083_rmse=direct.rmse(base,d['y'][q]),quarter_blend_rmse=direct.rmse(base+.25*eligible*(pred-base),d['y'][q]),replace_image_branch_rmse=direct.rmse(base+.16*eligible*(pred-p21),d['y'][q]),seconds=elapsed+time.monotonic()-start)
    direct.save(_repo_io(out/'done.json'),r);print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','train']);p.add_argument('--arm',choices=['micronet','imagenet'],default='micronet');p.add_argument('--tag',default='deep-kernel-micronet-f0');p.add_argument('--fold',type=int,default=0);p.add_argument('--seed',type=int,default=2026101911);p.add_argument('--steps',type=int,default=60);p.add_argument('--batch',type=int,default=8);p.add_argument('--lr',type=float,default=.0001);p.add_argument('--anchor',type=float,default=.0001);p.add_argument('--device',default='mps');a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):check_gradient() if a.action=='check' else run(a)
