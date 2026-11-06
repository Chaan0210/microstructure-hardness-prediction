import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from scipy.spatial.distance import pdist,cdist,squareform
from scipy.linalg import solve

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

def compact_existing(G,names):
    def g(n):return G[:,names.index(n)]
    cols=[.5*(g('grain_s1_interior_hall_petch')+g('grain_s2_interior_hall_petch')),g('grain_s1_interior_diam_std')/np.maximum(g('grain_s1_interior_diam_mean'),1.),g('phase_s1_0_fraction'),g('phase_s1_2_fraction'),g('pore_s1_t0.15_fraction'),g('pore_s1_t0.3_fraction'),g('pore_s1_t0.15_diam_mean'),g('grain_s1_all_aspect_mean'),g('grain_s1_alignment'),g('grain_s1_within_std_mean'),g('scale_robustness_diam'),g('grain_s1_boundary_fraction')]
    names=['inverse_sqrt_grain_diameter','grain_size_cv','phase_dark','phase_light','pore_fraction15','pore_fraction30','pore_diameter','grain_aspect','grain_alignment','within_grain_contrast','grain_scale_sensitivity','boundary_fraction']
    blocks=['size','size','phase','phase','phase','phase','phase','shape','shape','intensity','size','shape']
    return np.column_stack(cols),names,blocks
