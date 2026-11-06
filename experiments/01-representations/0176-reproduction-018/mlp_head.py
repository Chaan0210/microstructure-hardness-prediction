"""Own cached-feature dual-head MLP. No downloaded competition code."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler

CONFIG={'hidden':64,'dropout':0.25,'epochs':100,'batch_size':64,'lr':0.0003,'min_lr':0.00001,'weight_decay':0.01,'gradient_clip':1.0,'beta':1.0,'aux_weight':0.3,'bins':7,'seed':20261128}
class Head(nn.Module):
 def __init__(self, input_dim=3456):
  super().__init__();self.shared=nn.Sequential(nn.Linear(input_dim,64),nn.GELU(),nn.Dropout(.25));self.reg=nn.Linear(64,1);self.cls=nn.Linear(64,7)
 def forward(self,x):
  h=self.shared(x);return self.reg(h).squeeze(-1),self.cls(h)

def state_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def preprocess(X,y):
 scaler=StandardScaler().fit(X);mean=float(y.mean());std=float(y.std());assert std>0
 edges=np.quantile(y,np.arange(1,7)/7,method='linear');assert np.all(np.diff(edges)>0)
 bins=np.searchsorted(edges,y,side='right');assert set(bins)==set(range(7))
 return scaler,mean,std,edges,bins

def init_model(seed, input_dim=3456):
 torch.manual_seed(seed);return Head(input_dim)

def shared_gradient_check(X,y,bins,seed):
 m=init_model(seed).eval();r,c=m(X);lr=F.smooth_l1_loss(r,y,beta=1.);lc=F.cross_entropy(c,bins)
 params=list(m.shared.parameters());gr=torch.autograd.grad(lr,params,retain_graph=True);gc=torch.autograd.grad(.3*lc,params)
 gr=torch.cat([v.ravel() for v in gr]);gc=torch.cat([v.ravel() for v in gc]);assert torch.isfinite(gc).all() and gc.norm()>0
 return {'regression_loss_standardized':float(lr.detach()),'cross_entropy':float(lc.detach()),'weighted_cross_entropy':float(.3*lc.detach()),'shared_reg_gradient_norm':float(gr.norm()),'shared_weighted_class_gradient_norm':float(gc.norm()),'shared_gradient_cosine':float(torch.dot(gr,gc)/(gr.norm()*gc.norm()))}

def fit(X,y,bins,seed,aux_weight,epochs=None,validation=None):
 cfg=CONFIG;epochs=cfg['epochs'] if epochs is None else epochs
 m=init_model(seed, X.shape[1]);initial=state_hash(m.state_dict());initial_classifier={n:p.detach().clone() for n,p in m.cls.named_parameters()}
 opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'],eta_min=cfg['min_lr'])
 # Independent permutation generator makes data order identical between arms.
 orders=np.random.default_rng(seed+1).permuted(np.tile(np.arange(len(y)),(epochs,1)),axis=1)
 order_hash=hashlib.sha256(orders.tobytes()).hexdigest();torch.manual_seed(seed+1000)
 curves=[]
 def metrics(epoch):
  m.eval()
  with torch.no_grad():
   rp,cp=m(X);item={'epoch':epoch,'train_rmse_standardized':float(torch.mean((rp-y)**2).sqrt()),'train_smooth_l1':float(F.smooth_l1_loss(rp,y,beta=1.)),'train_ce':float(F.cross_entropy(cp,bins)),'train_class_accuracy':float((cp.argmax(1)==bins).float().mean())}
   if validation is not None:
    vx,vy,vb=validation;rv,cv=m(vx);item.update(validation_rmse_standardized=float(torch.mean((rv-vy)**2).sqrt()),validation_ce=float(F.cross_entropy(cv,vb)),validation_class_accuracy=float((cv.argmax(1)==vb).float().mean()))
  curves.append(item)
 metrics(0)
 for epoch in range(epochs):
  m.train()
  for start in range(0,len(y),cfg['batch_size']):
   ix=orders[epoch,start:start+cfg['batch_size']];opt.zero_grad(set_to_none=True);rp,cp=m(X[ix]);reg=F.smooth_l1_loss(rp,y[ix],beta=1.);loss=reg+aux_weight*F.cross_entropy(cp,bins[ix]) if aux_weight else reg
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['gradient_clip']);opt.step()
  scheduler.step()
  if (epoch+1)%20==0 or epoch+1==epochs:metrics(epoch+1)
 classifier_delta=max(float((p-initial_classifier[n]).abs().max()) for n,p in m.cls.named_parameters())
 assert classifier_delta>0 if aux_weight else classifier_delta==0
 return m.eval(),{'initial_state_sha256':initial,'batch_order_sha256':order_hash,'classifier_max_parameter_change':classifier_delta,'curves':curves,'final_state_sha256':state_hash(m.state_dict())}

def predict(m,X,mean,std):
 m.eval()
 with torch.inference_mode():r,_=m(torch.as_tensor(X,dtype=torch.float32));return r.numpy().astype(float)*std+mean
