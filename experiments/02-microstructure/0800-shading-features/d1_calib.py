"""D1: calibration / dispersion / W of the 039 dev OOF (4 variants), per partition."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys
E = str(_repo_path(_REPO_ROOT, 'experiments')) + '/'
files = [E + f'push-600/dev-{v}H2Z-angh.npz' for v in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4')]
D = [np.load(_repo_io(f), allow_pickle=True) for f in files]
F = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True)
keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
print('n cols', len(keys))
rm = lambda v, y: float(np.sqrt(np.mean((v - y) ** 2)))
for n in ['original', '20261031', '20261101']:
    ids = [str(i) for i in D[0][f'{n}_ids']]; y = D[0][f'{n}_y']; p = D[0][f'{n}_p021']; f = D[0][f'{n}_failed']
    g = np.mean([d[f'{n}_g'] for d in D], 0); g = np.where(f, p, g)
    b = 0.25 * p + 0.75 * g
    out = {}
    for name, v in (('g', g), ('p021', p), ('blend', b)):
        A = np.c_[np.ones_like(v), v]; coef = np.linalg.lstsq(A, y, rcond=None)[0]
        out[name] = dict(rmse=round(rm(v, y), 3), slope=round(coef[1], 3), sd=round(v.std(), 2), sd_opt=round(np.sqrt(max(y.var() - rm(v, y) ** 2, 0)), 2), mean_bias=round(float((v - y).mean()), 3))
    # W* and 2-coef OLS
    A = np.c_[np.ones_like(g), g, p]; c = np.linalg.lstsq(A, y, rcond=None)[0]
    wstar = float(np.sum((y - p) * (g - p)) / np.sum((g - p) ** 2))
    print(n, out, 'OLS(1,g,p)=', np.round(c, 3), 'W*', round(wstar, 3), 'corr(g-y,p-y)', round(np.corrcoef(g - y, p - y)[0, 1], 3))
# residual structure (original partition)
n = 'original'; ids = [str(i) for i in D[0][f'{n}_ids']]; y = D[0][f'{n}_y']; p = D[0][f'{n}_p021']; f = D[0][f'{n}_failed']
g = np.where(f, p, np.mean([d[f'{n}_g'] for d in D], 0)); b = 0.25 * p + 0.75 * g; r = y - g
X = F['X'][[pos[i] for i in ids]].astype(float)
def col(k): return X[:, keys.index(k)]
print([k for k in keys][:200])
for k in ('n_obj',):
    v = col(k); q = np.quantile(v, [0.2, 0.4, 0.6, 0.8]); t = np.digitize(v, q)
    print(k, [(int((t == j).sum()), round(rm(g[t == j], y[t == j]), 2), round(float(np.median(v[t == j])), 0)) for j in range(5)])
# label distribution at extremes: shrinkage
q = np.quantile(y, [0.1, 0.9])
print('low-y decile: mean(y)', round(y[y < q[0]].mean(), 1), 'mean(g)', round(g[y < q[0]].mean(), 1), '| high-y decile mean(y)', round(y[y > q[1]].mean(), 1), 'mean(g)', round(g[y > q[1]].mean(), 1))
q = np.quantile(g, [0.1, 0.9])
print('low-pred decile: mean(y)', round(y[g < q[0]].mean(), 1), 'mean(g)', round(g[g < q[0]].mean(), 1), '| high-pred decile mean(y)', round(y[g > q[1]].mean(), 1), 'mean(g)', round(g[g > q[1]].mean(), 1))
from scipy import stats
print('resid skew/kurt', stats.skew(r), stats.kurtosis(r))
