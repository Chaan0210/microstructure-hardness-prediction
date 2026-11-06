import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import Ridge
CANDIDATES=("none","ridge1","ridge01")
LAMBDAS={"ridge1":1.,"ridge01":.1}

def fit_model(X,r,blocks,kind):
 if kind=='none':return {'kind':'none','n_train':len(r)}
 bound=MinMaxScaler(clip=True).fit(X);B=bound.transform(X);scale=StandardScaler().fit(B);w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);Z=scale.transform(B)*w;rs=float(np.sqrt(np.mean(r*r)));rs=rs if rs>1e-12 else 1.;alpha=LAMBDAS[kind]*len(r);reg=Ridge(alpha=alpha,fit_intercept=False,solver='cholesky').fit(Z,r/rs)
 return {'kind':kind,'bound':bound,'scale':scale,'weights':w,'residual_scale':rs,'alpha':alpha,'regressor':reg,'n_train':len(r)}

def predict(m,X):
 if m['kind']=='none':return np.zeros(len(X))
 return m['regressor'].predict(m['scale'].transform(m['bound'].transform(X))*m['weights'])*m['residual_scale']
