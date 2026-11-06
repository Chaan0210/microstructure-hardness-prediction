"""GP (RBF+white on quantile-normal features) blended with cached TabPFN OOF on the same harness folds. usage: gp_blend.py FEAT_NPZ TABPFN_OOF_PREFIX"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
fn, pre = sys.argv[1], sys.argv[2]
F = np.load(_repo_io(fn), allow_pickle=True); X = F['X'].astype(float); ids = [str(i) for i in F['ids']]; y = harness.labels(ids); nz = np.load(_repo_io('../push-300/noise_train.npy'))
base = np.stack([np.load(_repo_io(f'{pre}_s{s}.npy')) for s in range(3)]).mean(0)
class GP:
    def fit(s, X, y):
        s.p = make_pipeline(SimpleImputer(strategy='median'), QuantileTransformer(n_quantiles=200, output_distribution='normal'), StandardScaler()); Z = s.p.fit_transform(X)
        s.mu = y.mean(); s.g = GaussianProcessRegressor(C(300.0) * RBF(8.0) + WhiteKernel(60.0), normalize_y=False, random_state=0).fit(Z, y - s.mu); return s
    def predict(s, X): return s.g.predict(s.p.transform(X)) + s.mu
O = harness.cv_predict(GP, X, y, reps=(0, 1, 2)); harness.report('gp', O, y, nz); harness.report('tabpfn', base, y, nz)
for a in (0.1, 0.2, 0.3): harness.report(f'tabpfn*{1-a:.1f}+gp*{a:.1f}', (1 - a) * base + a * O, y, nz)
