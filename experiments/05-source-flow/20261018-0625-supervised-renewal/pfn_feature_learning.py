"""Nested episodic learning of a shared feature map through frozen local TabPFN.

All context and training-query rows come from the fitting partition. Inner
selection rows and outer evaluation rows never enter feature-map learning.
ECDF ranks are piecewise constant: gradients follow smooth value paths locally;
finite differences across rank ties need not agree with those local gradients.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
import argparse,copy,hashlib,json,math,resource,time,gc
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import pfn_gradient_probe as pf

def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))
def rmse(p,y):return float(np.sqrt(np.mean((p-y)**2)))

class Features:
    def fit(self,X):
        self.median=np.nan_to_num(np.nanmedian(X,0));Z=np.where(np.isfinite(X),X,self.median);self.mean=Z.mean(0);self.scale=np.maximum(Z.std(0),.001);self.missing_columns=np.flatnonzero((~np.isfinite(X)).any(0));return self
    def transform(self,X):
        Z=np.where(np.isfinite(X),X,self.median);N=np.clip((Z-self.mean)/self.scale,-6,6);M=(~np.isfinite(X[:,self.missing_columns])).astype(np.float32);return torch.tensor(np.c_[N,M],dtype=torch.float32)
    def state(self):return {k:np.array(v) for k,v in vars(self).items()}

class Adapter(nn.Module):
    def __init__(self,p,numeric=104):
        super().__init__();self.numeric=numeric;self.down=nn.Linear(p,16);self.up=nn.Linear(16,numeric,bias=False);nn.init.zeros_(self.up.weight)
    def forward(self,x):
        delta=.1*self.up(torch.tanh(self.down(x)));return torch.cat([x[:,:self.numeric]+delta,x[:,self.numeric:]],1)

@torch.no_grad()
def predict(reg,model,X,y,Q):return pf.forward(reg,model(X),y,model(Q)).detach().cpu().numpy()

def fit_stage(a,d,X,train,query,out,steps,select=False,probe=False):
    out.mkdir(exist_ok=True);torch.manual_seed(a.seed);norm=Features().fit(X[train]);T=norm.transform(X[train]);Q=norm.transform(X[query]);y=torch.tensor(d['y'][train],dtype=torch.float32);model=Adapter(T.shape[1]);reg=pf.make(0);rng=np.random.default_rng(a.seed)
    spec=dict(train_ids=d['ids'][train].tolist(),query_ids=d['ids'][query].tolist(),steps=steps,select=select,seed=a.seed,horizon=a.horizon,context=a.context,queries=a.queries,rank=16,lr=a.lr,output_penalty=.1,weight_decay=.001,
        preprocessing='Train-only median imputation, mean/std,clip±6; append train-selected missingness flags unchanged. Identity-initialized104numeric lowrank16tanhmap; scaling0.1. No test/outer query fitting.',
        head='Existing local TabPFN3.5 frozen218982656weights; float32,oneestimator,seed0, differentiableidentitypreprocessor, fingerprintdisabled. Conditional on inherited physical104 representations.',
        selection='Innerqueryonly selectsamong0,25,50,100,200; refit onouterT withunchanged200stepLRhorizon. Outerquery neverusedforselection.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256((_repo_path(HERE, 'pfn_gradient_probe.py')).read_bytes()).hexdigest())
    save(out/'recipe.json',spec)
    initial=predict(reg,model,T,y,Q);np.savez(_repo_io(out/'initial.npz'),query=query,prediction=initial);curve=[dict(step=0,query_rmse=rmse(initial,d['y'][query]))];best=copy.deepcopy(model.state_dict());beststep=0;bestloss=curve[0]['query_rmse'];opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.001);start=time.monotonic();first=0
    if (out/'resume.pt').exists() and not probe:
        ck=torch.load(_repo_io(out/'resume.pt'),map_location='cpu',weights_only=False);assert ck['recipe']==spec;model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);first=ck['step'];rng.bit_generator.state=ck['rng'];curve=ck['curve'];best=ck['best'];beststep=ck['beststep'];bestloss=ck['bestloss']
    for step in range(first,steps):
        order=rng.permutation(len(train));nctx=min(a.context,len(train)-a.queries);c=order[:nctx];q=order[nctx:nctx+a.queries];assert not set(c)&set(q)
        opt.zero_grad(set_to_none=True);Z=model(T);pred=pf.forward(reg,Z[c],y[c],Z[q]);loss=((pred-y[q])/20).square().mean()+.1*(Z[:,:104]-T[:,:104]).square().mean();assert torch.isfinite(loss);loss.backward()
        grad=float(nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(grad) and grad>0
        memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if probe:
            r=dict(seconds=time.monotonic()-start,memory_bytes=memory,gradient_norm=grad,context=nctx,queries=len(q),optimizer_updates=0,local_model_frozen=all(not p.requires_grad and p.grad is None for m in reg.models_ for p in m.parameters()));save(out/'probe.json',r);print('PROBE',json.dumps(r),flush=True);return r
        assert memory<5*2**30
        for g in opt.param_groups:g['lr']=a.lr*(.1+.9*.5*(1+math.cos(math.pi*step/a.horizon)))
        opt.step()
        if step==0 or (step+1)%10==0:print('TRAIN',out.name,step+1,float(loss.detach()),'seconds',time.monotonic()-start,'rss',memory,flush=True)
        if step+1 in [25,50,100,200] or step+1==steps:
            pr=predict(reg,model,T,y,Q);score=rmse(pr,d['y'][query]);row=dict(step=step+1,query_rmse=score,seconds=time.monotonic()-start);curve.append(row);print('EVAL',out.name,json.dumps(row),flush=True)
            if select and score<bestloss:bestloss=score;beststep=step+1;best=copy.deepcopy(model.state_dict())
            save(out/'curve.json',curve)
        if (step+1)%10==0 or step+1==steps:
            ck=dict(state=model.state_dict(),optimizer=opt.state_dict(),recipe=spec,step=step+1,rng=rng.bit_generator.state,curve=curve,best=best,beststep=beststep,bestloss=bestloss);torch.save(ck,_repo_io(out/'resume-new.pt'));(out/'resume-new.pt').replace(out/'resume.pt');gc.collect()
    if select:model.load_state_dict(best)
    pred=predict(reg,model,T,y,Q);torch.save(dict(state=model.state_dict(),recipe=spec,normalization=norm.state(),selected_step=beststep if select else steps),_repo_io(out/'model.pt'));np.savez(_repo_io(out/'predictions.npz'),train=train,query=query,prediction=pred,initial=initial)
    result=dict(rmse=rmse(pred,d['y'][query]),initial_rmse=rmse(initial,d['y'][query]),selected_step=beststep if select else steps,seconds=time.monotonic()-start);save(out/'done.json',result);return result

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d,X=pf.load();tr=d['dev'][d['fold']!=a.fold];query=d['dev'][d['fold']==a.fold];A,B=train_test_split(tr,test_size=.2,random_state=2026101933+a.fold);A=np.sort(A);B=np.sort(B);out=_repo_path(HERE, 'runs', a.tag);out.mkdir(exist_ok=True)
    if a.action=='probe':return fit_stage(a,d,X,A,B,out/'probe',1,probe=True)
    selpath=out/'inner/done.json'
    if selpath.exists():sel=json.loads(selpath.read_text())
    else:sel=fit_stage(a,d,X,A,B,out/'inner',a.horizon,select=True)
    result=fit_stage(a,d,X,tr,query,out/'outer',sel['selected_step']);z=np.load(_repo_io(out/'outer/predictions.npz'));ix=np.flatnonzero(d['fold']==a.fold);base=d['base083'][ix];eligible=~d['failed'][ix];pred=z['prediction'];initial=z['initial'];system=base+.384*eligible*(pred-initial)
    result.update(base083_rmse=rmse(base,d['y'][query]),system_rmse=rmse(system,d['y'][query]),primary='Complete083+.384*(learned-feature prediction -matchedidentity control), existingfailurefallback.')
    np.savez(_repo_io(out/'predictions.npz'),train=tr,query=query,prediction=pred,control=initial,system=system);save(out/'done.json',result);print('DONE',json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['probe','train']);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--tag',default='pfn-feature-f0');ap.add_argument('--seed',type=int,default=2026101931);ap.add_argument('--horizon',type=int,default=200);ap.add_argument('--context',type=int,default=192);ap.add_argument('--queries',type=int,default=32);ap.add_argument('--lr',type=float,default=.003)
    with threadpool_limits(2):main(ap.parse_args())
