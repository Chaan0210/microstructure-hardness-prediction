import common as c
from common import np

def geometry(X,T):
 mu=[];var=[];sc=[];rms=[]
 for j in range(X.shape[1]//768):
  q=X[T,j*768:(j+1)*768];m=q.mean(0);v=q.var(0);r=float(np.sqrt(v.mean()));mu.append(m);var.append(v);sc.append(np.maximum(np.sqrt(v),r) if r>0 else np.ones_like(v));rms.append(r)
 g={'mean':np.concatenate(mu),'variance':np.concatenate(var),'scale':np.concatenate(sc),'bank_RMS':np.array(rms),'train':T.copy()};A=(X-g['mean'])/g['scale'];g['center']=A[T].mean(0);Z=A-g['center'];U,s,V=np.linalg.svd(Z[T],full_matrices=False);assert len(s)==320 and np.isfinite(s).all();g['singular_values']=s
 return g,Z,U,V

def alpha_for_df(lam):
 lo=0.;hi=float(lam.max());assert hi>0
 while float(np.sum(lam/(lam+hi)))>32:hi*=2
 for _ in range(120):
  mid=(lo+hi)/2;value=float(np.sum(lam/(lam+mid)))
  if value>32:lo=mid
  else:hi=mid
 alpha=(lo+hi)/2;df=float(np.sum(lam/(lam+alpha)));assert alpha>0 and abs(df-32)<1e-10;return alpha,df

def fit(g,U,V,y,T):
 s=g['singular_values'];alpha,df=alpha_for_df(s*s);ym=float(y[T].mean());ys=float(y[T].std());assert ys>0;coef=V.T@((s/(s*s+alpha))*(U.T@((y[T]-ym)/ys)));assert np.isfinite(coef).all()
 return {**g,'coefficient':coef,'alpha':np.array(alpha),'EDF':np.array(df),'target_mean':np.array(ym),'target_std':np.array(ys)}
def transform(X,m):return (X-m['mean'])/m['scale']-m['center']
def predict(Z,m):return (Z@m['coefficient'])*float(m['target_std'])+float(m['target_mean'])
def diagnostics(g,Z,U,alpha,T,V):
 s=g['singular_values'];lam=s*s;threshold=np.finfo(float).eps*max(len(T),Z.shape[1])*s[0];rank=int((s>threshold).sum());proportion=lam/lam.sum();hat=lam/(lam+alpha);leverage=(U*U)@hat
 bank=[]
 for j in range(Z.shape[1]//768):
  q=Z[:,j*768:(j+1)*768];bank.append({'index':j,'RMS_floor':float(g['bank_RMS'][j]),'train_abs_quantiles':np.quantile(abs(q[T]),[.5,.95,.99,1]).tolist(),'validation_abs_quantiles':np.quantile(abs(q[V]),[.5,.95,.99,1]).tolist(),'validation_fraction_abs_gt10':float(np.mean(abs(q[V])>10))})
 return {'numerical_rank_diagnostic_only':rank,'threshold':float(threshold),'nonzero_condition':float(s[0]/s[rank-1]),'eigen_energy_fraction':{str(n):float(proportion[:n].sum()) for n in [1,10,32,128]},'entropy_effective_rank':float(np.exp(-np.sum(proportion[proportion>0]*np.log(proportion[proportion>0])))),'df_beyond128':float(hat[128:].sum()),'train_leverage_quantiles':np.quantile(leverage,[0,.5,.95,1]).tolist(),'train_leverage_sum':float(leverage.sum()),'bank_scale_diagnostics':bank}
