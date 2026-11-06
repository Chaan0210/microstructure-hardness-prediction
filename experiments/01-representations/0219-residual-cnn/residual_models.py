import numpy as np
from scipy.spatial.distance import pdist,cdist,squareform
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import residual_linear as r
KINDS=(*r.CANDIDATES,"rbf1","rbf01")
LAM={"rbf1":1.,"rbf01":.1}

def fit(X,y,blocks,kind):
 if kind in r.CANDIDATES:return r.fit_model(X,y,blocks,kind)
 bound=MinMaxScaler(clip=True).fit(X);sc=StandardScaler().fit(bound.transform(X));w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);Z=sc.transform(bound.transform(X))*w;dist=pdist(Z,'sqeuclidean');positive=dist[dist>1e-12];gamma=1/float(np.median(positive)) if len(positive) else 1.;K=np.exp(-gamma*squareform(dist));col=K.mean(0);grand=float(K.mean());C=K-K.mean(1)[:,None]-col[None,:]+grand;scale=float(np.trace(C)/len(C));degenerate=scale<=1e-12;scale=1. if degenerate else scale;C=C/scale;rs=float(np.sqrt(np.mean(y*y)));rs=rs if rs>1e-12 else 1.;alpha=LAM[kind]*len(y);coef=np.linalg.solve(C+alpha*np.eye(len(y)),y/rs);eigen=np.linalg.eigvalsh(C);df=float(np.sum(eigen/(eigen+alpha)));assert df<=1/LAM[kind]+1e-8
 return {'kind':kind,'n_train':len(y),'bound':bound,'scale':sc,'weights':w,'Z':Z,'gamma':gamma,'column':col,'grand':grand,'kernel_scale':scale,'degenerate':degenerate,'residual_scale':rs,'alpha':alpha,'coef':coef,'effective_df':df}

def predict(m,X):
 if m['kind'] in r.CANDIDATES:return r.predict(m,X)
 Z=m['scale'].transform(m['bound'].transform(X))*m['weights'];K=np.exp(-m['gamma']*cdist(Z,m['Z'],'sqeuclidean'));C=(K-K.mean(1)[:,None]-m['column'][None,:]+m['grand'])/m['kernel_scale'];return C@m['coef']*m['residual_scale']
