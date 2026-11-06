import common as c
from common import np

def geometry(X,T):
 means=[];variances=[];scales=[];floors=[]
 for j in range(2):
  q=X[T,j*768:(j+1)*768];mu=q.mean(0);var=q.var(0);r=float(np.sqrt(var.mean()));sc=np.maximum(np.sqrt(var),r) if r>0 else np.ones_like(var);means.append(mu);variances.append(var);scales.append(sc);floors.append(r)
 mu=np.concatenate(means);scale=np.concatenate(scales);A=(X-mu)/scale;center=A[T].mean(0);Z=A-center;U,s,V=np.linalg.svd(Z[T],full_matrices=False);assert len(s)==320 and np.isfinite(s).all();assert (s>np.finfo(float).eps*1536*s[0]).sum()>128
 return {'raw_mean':mu,'raw_var':np.concatenate(variances),'raw_scale':scale,'bank_RMS':np.array(floors),'center':center,'U':U,'singular_values':s,'V':V,'train':T.copy()},Z

def transform(X,g):return (X-g['raw_mean'])/g['raw_scale']-g['center']
def alpha_for_df(eigenvalues):
 target=32.;lo=0.;hi=float(eigenvalues.max())
 while float(np.sum(eigenvalues/(eigenvalues+hi)))>target:hi*=2
 for _ in range(120):
  mid=(lo+hi)/2;df=float(np.sum(eigenvalues/(eigenvalues+mid)))
  if df>target:lo=mid
  else:hi=mid
 alpha=(lo+hi)/2;df=float(np.sum(eigenvalues/(eigenvalues+alpha)));assert alpha>0 and abs(df-target)<1e-10;return alpha,df

def fit(g,y,T,arm):
 k=128 if arm=='pca128_ridge' else len(g['singular_values']);s=g['singular_values'][:k];V=g['V'][:k];U=g['U'][:,:k];alpha,df=alpha_for_df(s*s);ym=float(y[T].mean());ys=float(y[T].std());yt=(y[T]-ym)/ys;coef=V.T@((s/(s*s+alpha))*(U.T@yt));assert np.isfinite(coef).all()
 return {'coef':coef,'alpha':np.array(alpha),'effective_df':np.array(df),'target_mean':np.array(ym),'target_std':np.array(ys),'retained_singular_directions':np.array(k),'train':T.copy()}
def predict(Z,o):return (Z@o['coef'])*float(o['target_std'])+float(o['target_mean'])
