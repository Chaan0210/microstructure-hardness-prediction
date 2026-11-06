"""Constrained residual fusion, with all nuisance fits inside the training scope."""
import common as c
import numpy as np
from scipy.optimize import minimize,LinearConstraint
ns=c.definitions(c.R4/'noise_calibration.py',['fit_weights','select'],dict(np=np,STRENGTHS=('global',100,30,10,0)))
CONFIGS=[('none',0),('fixed',0)]+[(a,l) for a in ['shared','global','conditional'] for l in [100,10,1]]
def subset(d,ix,labels=True):return {k:np.asarray(v)[ix] for k,v in d.items() if labels or k!='y'}
def high(d):return (d['noise']>=12)&d['eligible']
def incumbent_beta(d):
 good=d['eligible'];w,_=ns['select'](d['R'][good,0],(d['y']-d['p73'])[good],d['noise'][good]>=12);return float(w[1])
def quality_fit(x):
 mu=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-8,1.,sd);return mu,sd
def basis(x,mu,sd,conditional):return np.c_[np.ones(len(x)),np.clip((x-mu)/sd,-2,2)] if conditional else np.ones((len(x),1))
def matrix(R,U):return (R[:,:,None]*U[:,None,:]).reshape(len(R),-1)
def constraints(m,width,beta):
 corners=np.array([[1.,x,y] for x in [-2.,2.] for y in [-2.,2.]]) if width==3 else np.ones((1,1));rows=[];low=[];upper=[]
 for u in corners:
  for j in range(m):
   row=np.zeros((m,width));row[j]=u;rows.append(row.ravel());low.append(beta/2 if j==0 else 0.);upper.append(.5 if j==4 else 1.)
  row=np.zeros((m,width));row[:min(m,4)]=u;rows.append(row.ravel());low.append(0.);upper.append(1.)
 return np.array(rows),np.array(low),np.array(upper)
def solve(R,Q,residual,beta,kind,lam):
 conditional=kind=='conditional';m=1 if kind=='shared' else 5;R=R[:,:m];mu,sd=quality_fit(Q);U=basis(Q,mu,sd,conditional);width=U.shape[1];J=matrix(R,U);theta0=np.zeros((m,width));theta0[0,0]=beta;theta0=theta0.ravel();active=np.sqrt(np.mean(J**2,0))>1e-8;epsilon=1e-8*max(float(np.mean(R**2)),1.);Jint=J.copy()
 if conditional:Jint[:,np.arange(J.shape[1])%3==0]=0
 else:Jint[:]=0
 H=(1+lam)*(J.T@J)/len(J)+(10*lam)*(Jint.T@Jint)/len(J)+epsilon*np.eye(J.shape[1]);b=J.T@residual/len(J);A,lb,ub=constraints(m,width,beta);theta=theta0.copy();nit=0
 if kind=='shared':
  change=float(b[0]/H[0,0]) if active[0] else 0.;theta[0]=np.clip(beta+change,beta/2,1.)
 elif active.any():
  aa=A[:,active];lo=lb-A@theta0;up=ub-A@theta0;keep=np.linalg.norm(aa,axis=1)>1e-12;assert np.all(lo[~keep]<=1e-10) and np.all(up[~keep]>=-1e-10);h=H[np.ix_(active,active)];bb=b[active]
  fun=lambda w:float(w@h@w-2*bb@w)
  jac=lambda w:2*(h@w-bb)
  result=minimize(fun,np.zeros(active.sum()),jac=jac,method='SLSQP',constraints=[LinearConstraint(aa[keep],lo[keep],up[keep])],options=dict(ftol=1e-10,maxiter=300));assert result.success,(kind,lam,result.message);theta[active]+=result.x;nit=int(result.nit)
 violation=max(float(np.max(lb-A@theta)),float(np.max(A@theta-ub)),0.);assert violation<1e-7,(kind,lam,violation)
 return dict(kind=kind,lam=lam,beta=beta,theta=theta.reshape(m,width).tolist(),quality_mean=mu.tolist(),quality_scale=sd.tolist(),epsilon=epsilon,n_high=len(R),constraint_violation=violation,iterations=nit,active=active.tolist())
def fit(d,kind,lam,beta=None):
 beta=incumbent_beta(d) if beta is None else beta;h=high(d);R=d['R'][h];residual=(d['y']-d['p76']-beta*d['R'][:,0])[h]
 if kind=='none':return dict(kind=kind,beta=beta)
 if kind=='fixed':
  g=R[:,4];a=float(np.clip(g@residual/(g@g+len(g)),0,.5));return dict(kind=kind,beta=beta,topology_beta=a)
 assert len(R)>=3
 return solve(R,d['quality'][h],residual,beta,kind,lam)
def predict(model,q):
 assert 'y' not in q,'Query labels are not a prediction capability';h=high(q);p=q['p76'].copy();R=q['R'][h];kind=model['kind'];beta=model['beta']
 if kind=='none':add=beta*R[:,0]
 elif kind=='fixed':add=beta*(.625*R[:,0]+.125*R[:,1:4].sum(1))+.125*model['topology_beta']*R[:,4]
 else:
  U=basis(q['quality'][h],np.array(model['quality_mean']),np.array(model['quality_scale']),kind=='conditional');theta=np.array(model['theta']);coef=U@theta.T;assert coef.min(initial=0)>-1e-7 and np.max(coef[:,:min(4,len(theta))].sum(1),initial=0)<=1+1e-7
  add=(R[:,:len(theta)]*coef).sum(1)
 p[h]+=add;assert np.isfinite(p).all();return p
def select_all(d):
 hi=np.flatnonzero(high(d));pred=np.empty((len(CONFIGS),len(hi)));betas=[];maxviolation=0.;maxiter=0
 for t,j in enumerate(hi):
  ix=np.arange(len(d['y']))!=j;train=subset(d,ix);q=subset(d,np.array([j]),labels=False);beta=incumbent_beta(train);betas.append(beta)
  for k,(kind,lam) in enumerate(CONFIGS):
   model=fit(train,kind,lam,beta);pred[k,t]=predict(model,q)[0];maxviolation=max(maxviolation,model.get('constraint_violation',0));maxiter=max(maxiter,model.get('iterations',0))
 error=(pred-d['y'][hi])**2;loss=error.mean(1)
 def choose(which):
  allowed=[j for j,z in enumerate(CONFIGS) if z[0] in which];best=min(loss[j] for j in allowed);return next(j for j in allowed if loss[j]<=best+1e-10)
 selected={name:choose(names) for name,names in dict(primary=['none','fixed','shared','global','conditional'],shared=['none','shared'],global_=['none','global'],conditional=['none','conditional']).items()};selected['fixed']=1;selected['none']=0
 beta=incumbent_beta(d);models={str(j):fit(d,*CONFIGS[j],beta) for j in sorted(set(selected.values()))}
 return dict(configs=[list(z) for z in CONFIGS],loss=loss.tolist(),selected=selected,models=models,full_beta=beta,high_rows=hi.tolist(),loo_beta=betas,loo_predictions=pred.tolist(),max_constraint_violation=maxviolation,max_optimizer_iterations=maxiter)
