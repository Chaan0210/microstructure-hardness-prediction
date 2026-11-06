"""Exact BalancedKernel class extracted from existing compact_regression.py for isolated environment compatibility."""
import numpy as np
from scipy.spatial.distance import pdist,cdist,squareform
from scipy.linalg import solve
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class BalancedKernel(RegressorMixin,BaseEstimator):
    def __init__(self,blocks=None,bandwidth=1.,alpha=.1):self.blocks=blocks;self.bandwidth=bandwidth;self.alpha=alpha
    def fit(self,X,y):
        self.bound_=MinMaxScaler(clip=True).fit(X);self.scale_=StandardScaler().fit(self.bound_.transform(X))
        blocks=np.array(self.blocks if self.blocks is not None else [0]*X.shape[1]);self.weights_=np.array([1/np.sqrt(np.sum(blocks==b)) for b in blocks])
        self.X_=self.scale_.transform(self.bound_.transform(X))*self.weights_
        dist=pdist(self.X_,'sqeuclidean');self.gamma_=1/max(float(np.median(dist[dist>1e-12]))*self.bandwidth**2,1e-12)
        self.ym_=float(np.mean(y));self.ys_=max(float(np.std(y)),1e-8);K=np.exp(-self.gamma_*squareform(dist));K.flat[::len(y)+1]+=self.alpha
        self.coef_=solve(K,(y-self.ym_)/self.ys_,assume_a='pos');self.n_features_in_=X.shape[1];return self
    def predict(self,X):
        z=self.scale_.transform(self.bound_.transform(X))*self.weights_;return np.exp(-self.gamma_*cdist(z,self.X_,'sqeuclidean'))@self.coef_*self.ys_+self.ym_
