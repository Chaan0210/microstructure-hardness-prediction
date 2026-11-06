"""Exploratory (descriptive) look at GP-ARD on v3 features, original split only. Not used for adoption by itself."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys, warnings, time, json
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from models_v3 import make_transform
F = np.load(_repo_io('features-v3-train.npz')); keys = list(F['keys']); pos = {i: k for k, i in enumerate(F['ids'])}
o = np.load(_repo_io('oof-v3.npz')); a = 'B_v3feat_v3mask_original_'; ids = o[a + 'ids']; y = o[a + 'y']; gk = o[a + 'g']; b = o[a + 'b']; p = 2 * b - gk
d = np.load(_repo_io('../resolution-216/ensemble-OOF.npz')); fold = d['fold']; assert (d['ids'] == ids).all()
X = F['X'][[pos[i] for i in ids]]; failed = np.nan_to_num(X[:, keys.index('n_obj')], nan=0) < 20
def gp():
    k = C(1.0) * RBF(length_scale=np.full(X.shape[1], 5.0), length_scale_bounds=(0.3, 1e3)) + WhiteKernel(0.5, (1e-3, 2))
    return make_pipeline(SimpleImputer(strategy='median'), make_transform(keys), StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=0))
g = np.zeros(len(y)); t0 = time.time(); ls = []
for k in range(5):
    tr, va = fold != k, fold == k; mu = y[tr].mean(); m = gp().fit(X[tr], y[tr] - mu); g[va] = m.predict(X[va]) + mu
    ls.append(m[-1].kernel_.k1.k2.length_scale)
g = np.where(failed, p, g); rm = lambda v: float(np.sqrt(np.mean((v - y) ** 2)))
L = np.median(np.array(ls), 0); top = [keys[i] for i in np.argsort(L)[:12]]
print(json.dumps(dict(gp_standalone=rm(g), krr_standalone=rm(gk), gp_blend=rm(0.5 * p + 0.5 * g), krr_blend=rm(b), gp_krr_mean_blend=rm(0.5 * p + 0.25 * g + 0.25 * gk), sec=round(time.time() - t0), most_relevant=top), indent=1))
