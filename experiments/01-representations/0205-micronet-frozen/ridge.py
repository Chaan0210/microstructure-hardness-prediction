from common import np

def geometry(X,T):
 q=X[T].astype(np.float64);mu=q.mean(0);var=q.var(0);floor=float(np.sqrt(var.mean()));scale=np.maximum(np.sqrt(var),floor) if floor>0 else np.ones_like(var)
 A=(q-mu)/scale;center=A.mean(0);Z=A-center;U,s,V=np.linalg.svd(Z,full_matrices=False)
 rank=int(np.sum(s>np.finfo(float).eps*max(Z.shape)*s[0]));assert rank>32
 lam=s*s;lo=0.;hi=float(lam.max())
 while np.sum(lam/(lam+hi))>32:hi*=2
 for _ in range(120):
  a=(lo+hi)/2
  if np.sum(lam/(lam+a))>32:lo=a
  else:hi=a
 alpha=(lo+hi)/2;df=float(np.sum(lam/(lam+alpha)));assert abs(df-32)<1e-8
 return {'mean':mu,'var':var,'scale':scale,'floor':np.array(floor),'center':center,'U':U,'s':s,'V':V,'alpha':np.array(alpha),'df':np.array(df),'rank':np.array(rank),'train':T}
def transform(X,g):return (X.astype(np.float64)-g['mean'])/g['scale']-g['center']
def fit(g,y,T):
 ym=float(y[T].mean());ys=float(y[T].std());assert ys>0;yt=(y[T]-ym)/ys;s=g['s'];alpha=float(g['alpha']);coef=g['V'].T@((s/(s*s+alpha))*(g['U'].T@yt));assert np.isfinite(coef).all()
 return {'coef':coef,'y_mean':np.array(ym),'y_std':np.array(ys),'alpha':g['alpha'],'df':g['df'],'train':T}
def predict(X,g,m):return transform(X,g)@m['coef']*float(m['y_std'])+float(m['y_mean'])
