"""Feature-space mixup alpha0.4, otherwise same051 optimizer/seeds/schedule."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
import auxiliary_head_051 as h
EPOCHS=(5,15,40,100)
SEEDS=(20261128,20261205,20261219)
ALPHA=.4
def mix(X,y,ix,partner,lam):
 assert len(ix)==len(partner)==len(lam) and np.all((lam>=0)&(lam<=1));assert set(partner.tolist())==set(ix.tolist())
 w=torch.tensor(lam,dtype=X.dtype)
 return w[:,None]*X[ix]+(1-w[:,None])*X[partner],w*y[ix]+(1-w)*y[partner]
def fit(X,y,seed):
 cfg=h.CONFIG;m=h.init_model(seed);initial=h.state_hash(m.state_dict());classifier={n:p.detach().clone() for n,p in m.cls.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=cfg['min_lr']);orders=np.random.default_rng(seed+1).permuted(np.tile(np.arange(len(y)),(100,1)),axis=1);torch.manual_seed(seed+1000);states={};curves=[];mixrng=np.random.default_rng(seed+2000);mixhash=hashlib.sha256()
 for epoch in range(1,101):
  m.train()
  for start in range(0,len(y),cfg['batch_size']):
   ix=orders[epoch-1,start:start+cfg['batch_size']];partner=mixrng.permutation(ix);lam=mixrng.beta(ALPHA,ALPHA,size=len(ix));mixhash.update(partner.tobytes());mixhash.update(lam.tobytes());xm,ym=mix(X,y,ix,partner,lam);opt.zero_grad(set_to_none=True);rp,_=m(xm);loss=F.smooth_l1_loss(rp,ym,beta=cfg['beta']);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['gradient_clip']);opt.step()
  sched.step()
  if epoch in EPOCHS:
   m.eval()
   with torch.no_grad():p,_=m(X);curves.append({'epoch':epoch,'train_rmse_standardized':float(torch.mean((p-y)**2).sqrt()),'train_loss':float(F.smooth_l1_loss(p,y,beta=cfg['beta']))})
   states[epoch]={n:p.detach().clone() for n,p in m.state_dict().items()}
 assert all(torch.equal(classifier[n],p.detach()) and p.grad is None for n,p in m.cls.named_parameters())
 return states,{'initial_hash':initial,'order_hash':hashlib.sha256(orders.tobytes()).hexdigest(),'curves':curves,'final_hash':h.state_hash(states[100]),'mixup_seed':seed+2000,'mixup_hash':mixhash.hexdigest()}
def predict(state,X,mean,std):
 m=h.Head();m.load_state_dict(state);return h.predict(m,X,mean,std)
