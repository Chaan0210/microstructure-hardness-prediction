"""Regressor diversity on the 030 feature set, harness folds (500 rows, 5-fold x reps 0,1,2). Baseline = cached TabPFN-v2 n8 OOF (seeds 0/1/2)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
F = np.load(_repo_io('../push-300/feat-a1s250n4vstd-stcoh-train.npz'), allow_pickle=True); X = F['X'].astype(float); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
y = harness.labels(ids); nz = np.load(_repo_io('../push-300/noise_train.npy'))
base = np.stack([np.load(_repo_io(f'../push-300/oof_none_v2_feat-a1s250n4vstd-stcoh-train_s{s}.npy')) for s in range(3)]).mean(0)
harness.report('tabpfn v2 n8 (3-seed mean)', base, y, nz)
which = sys.argv[1:]
def mk(name):
    if name == 'lgb':
        return lambda: lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.01, num_leaves=7, min_child_samples=10, subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1)
    if name == 'lgb2':
        return lambda: lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.005, max_depth=2, num_leaves=4, min_child_samples=8, subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=2, verbose=-1)
    if name == 'et':
        return lambda: make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(n_estimators=800, max_features=0.33, min_samples_leaf=2, n_jobs=-1, random_state=0))
    if name == 'gp':
        k = C(300.0) * RBF(length_scale=8.0) + WhiteKernel(60.0)
        class GP:
            def fit(s, X, y):
                s.p = make_pipeline(SimpleImputer(strategy='median'), QuantileTransformer(n_quantiles=200, output_distribution='normal'), StandardScaler()); Z = s.p.fit_transform(X)
                s.mu = y.mean(); s.g = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=0, random_state=0).fit(Z, y - s.mu); return s
            def predict(s, X): return s.g.predict(s.p.transform(X)) + s.mu
        return GP
O = {}
for w in which:
    t0 = time.time(); O[w] = harness.cv_predict(mk(w), X, y, reps=(0, 1, 2)); np.save(_repo_io(f'oof_div_{w}.npy'), O[w])
    harness.report(w + f' ({time.time()-t0:.0f}s)', O[w], y, nz)
    for a in (0.1, 0.2, 0.3):
        harness.report(f'  tabpfn*{1-a:.1f} + {w}*{a:.1f}', (1 - a) * base + a * O[w], y, nz)
    print('  resid corr with tabpfn', np.mean([np.corrcoef(O[w][r] - y, base[r] - y)[0, 1] for r in range(3)]).round(3), flush=True)
