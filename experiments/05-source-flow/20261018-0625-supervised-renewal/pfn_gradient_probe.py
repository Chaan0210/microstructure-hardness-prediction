"""Local, no-update differentiation/resource check on original training rows."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME']=str(_repo_path(HERE, 'tmp/huggingface'));os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import json,time,resource,gc
import numpy as np
import torch
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness

def make(seed=0):
    reg=harness.tabpfn_new('v3.5',seed,'cpu',n_est=1);reg.differentiable_input=True;reg.inference_precision=torch.float32;reg.fit_mode='fit_preprocessors';reg.inference_config={'FINGERPRINT_FEATURE':False};return reg

def forward(reg,X,y,Q):
    reg.fit_with_differentiable_input(X,y)
    for model in reg.models_:model.eval();model.requires_grad_(False)
    logits,_,_=reg.forward(Q,use_inference_mode=True);logits=logits.T
    return reg.znorm_space_bardist_.mean(logits)*reg.y_train_std_+reg.y_train_mean_

def load():
    d=np.load(_repo_path(HERE, 'data.npz'));fp=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(fp));pos={str(v):j for j,v in enumerate(z['ids'])};X=z['X'][[pos[str(v)] for v in d['ids']]].astype(np.float32);return d,X

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d,X=load();ix=d['dev'][d['fold']!=0];X=X[ix[:40]];med=np.nanmedian(X[:32],0);med=np.nan_to_num(med);X=np.where(np.isfinite(X),X,med);mu=X[:32].mean(0);sd=np.maximum(X[:32].std(0),.001);X=np.clip((X-mu)/sd,-6,6)
    x=torch.tensor(X,requires_grad=True);y=torch.tensor(d['y'][ix[:32]],dtype=torch.float32);truth=torch.tensor(d['y'][ix[32:40]],dtype=torch.float32)
    reg=make();start=time.monotonic();p=forward(reg,x[:32],y,x[32:]);loss=((p-truth)/20).square().mean();loss.backward();seconds=time.monotonic()-start
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(param.grad is None for model in reg.models_ for param in model.parameters())
    # Actual train-input finite-difference directional check, no optimization.
    direction=x.grad.detach()/x.grad.norm();eps=.001
    with torch.no_grad():
        plus=x.detach()+eps*direction;minus=x.detach()-eps*direction
        lp=((forward(reg,plus[:32],y,plus[32:])-truth)/20).square().mean();lm=((forward(reg,minus[:32],y,minus[32:])-truth)/20).square().mean()
        expected=float((x.grad*direction).sum());actual=float((lp-lm)/(2*eps));rel=abs(actual-expected)/max(abs(expected),1e-6)
    result=dict(seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,context_gradient_norm=float(x.grad[:32].norm()),query_gradient_norm=float(x.grad[32:].norm()),directional_autograd=expected,directional_finite_difference=actual,relative_error=rel,optimizer_updates=0,only_outer_training_ids=d['ids'][ix[:40]].tolist(),frozen_model_parameter_count=sum(p.numel() for m in reg.models_ for p in m.parameters()))
    (_repo_path(HERE, 'pfn-gradient-probe.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);assert rel<.05

if __name__=='__main__':
    with threadpool_limits(2):main()
