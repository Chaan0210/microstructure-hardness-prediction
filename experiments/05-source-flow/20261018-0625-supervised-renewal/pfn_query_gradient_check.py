from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import sys,json
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p));import pfn_gradient_probe as m
import torch,numpy as np
from threadpoolctl import threadpool_limits
torch.set_num_threads(2)
with threadpool_limits(2):
 d,X=m.load();ix=d['dev'][d['fold']!=0][:40];X=X[ix];med=np.nan_to_num(np.nanmedian(X[:32],0));X=np.where(np.isfinite(X),X,med);mu=X[:32].mean(0);sd=np.maximum(X[:32].std(0),.001);X=np.clip((X-mu)/sd,-6,6)
 x=torch.tensor(X,requires_grad=True);y=torch.tensor(d['y'][ix[:32]],dtype=torch.float32);truth=torch.tensor(d['y'][ix[32:40]],dtype=torch.float32);reg=m.make();pred=m.forward(reg,x[:32],y,x[32:]);loss=((pred-truth)/20).square().mean();loss.backward()
 margin=np.min(abs(X[32:,None]-X[None,:32]),axis=1);scores=x.grad[32:].abs().numpy()*(margin>.01);row,col=np.unravel_index(scores.argmax(),scores.shape);expected=float(x.grad[row+32,col]);reports=[]
 with torch.no_grad():
  for eps in [.001,.003,.01]:
   plus=x.detach().clone();minus=plus.clone();plus[row+32,col]+=eps;minus[row+32,col]-=eps;lp=((m.forward(reg,plus[:32],y,plus[32:])-truth)/20).square().mean();lm=((m.forward(reg,minus[:32],y,minus[32:])-truth)/20).square().mean();actual=float((lp-lm)/(2*eps));reports.append(dict(eps=eps,expected=expected,actual=actual,relative_error=abs(actual-expected)/abs(expected)))
 result=dict(row=int(row),column=int(col),margin=float(margin[row,col]),reports=reports);print(result);(_repo_path(p, 'pfn-query-gradient-check.json')).write_text(json.dumps(result,indent=2))
