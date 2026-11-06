"""Strict check of the qm alignment for the Ridge partners: band edges and per-band robust maps fitted on the outer-training rows of each fold only
(the 100 non-dev train rows excluded too), applied to the held-out fold. Same models as ridge_qm.py. usage: ridge_qm_strict.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import warnings, numpy as np
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from blend_dev import load, rmse, parts
B = load(9); ids, y, FO = B['ids'], B['y'], B['fold']
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
def X_of(path): F = np.load(_repo_io(path), allow_pickle=True); pos = {str(i): k for k, i in enumerate(F['ids'])}; return F['X'][[pos[i] for i in ids]].astype(float), [str(k) for k in F['keys']]
O = {v: X_of(f'../push-800/feat-{v}-anghcor-train.npz')[0] for v in V}; keys = X_of('../push-800/feat-u1dmavgH2Z-anghcor-train.npz')[1]; j = keys.index('noise')
def rs(x):
    x = x[np.isfinite(x)]
    if len(x) < 5: return np.nan, np.nan
    q1, q2, q3 = np.quantile(x, [.25, .5, .75]); return q2, (q3 - q1) / 1.349
def aligned(X, fit_rows):
    nzf = X[fit_rows, j]; fill = np.nanmedian(nzf); nzall = np.where(np.isnan(X[:, j]), fill, X[:, j]); med = np.median(np.where(np.isnan(nzf), fill, nzf))
    edges = np.quantile(nzall[fit_rows][nzall[fit_rows] >= med], [1 / 3, 2 / 3]); b = np.where(nzall < med, -1, np.digitize(nzall, edges)); out = X.copy()
    for c in range(X.shape[1]):
        if c == j: continue
        mr, sr = rs(X[fit_rows & (b == -1), c])
        for k in range(3):
            mb, sb = rs(X[fit_rows & (b == k), c])
            if not (np.isfinite(sb) and sb > 1e-12 and np.isfinite(sr)): continue
            m = b == k; out[m, c] = mr + (X[m, c] - mb) * sr / sb
    return out
r1 = np.zeros((3, 400)); l1 = np.zeros((3, 400))
for k in range(3):
    for f in range(5):
        tr, va = FO[k] != f, FO[k] == f; XV = {v: aligned(O[v], tr) for v in V}
        r1[k, va] = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(XV['u1dmavgH2Z'][tr], y[tr]).predict(XV['u1dmavgH2Z'][va])
        l1[k, va] = np.exp(np.mean([make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(XV[v][tr], np.log(y[tr])).predict(XV[v][va]) for v in V], 0))
fail, p = B['fail'], B['p021']
def c54(r, l): c51 = 0.2 * p + 0.48 * B['g'] + 0.16 * r + 0.16 * l; return np.where(fail, p, 0.77 * c51 + 0.08 * B['set'] + 0.15 * B['gmm'])
base = c54(B['ridge'], B['logridge']); c = c54(r1, l1)
print(f'strict: ridge alone {rmse(np.where(fail, p, r1), y):.3f} logridge alone {rmse(np.where(fail, p, l1), y):.3f}  054 {rmse(c, y):.4f} ({rmse(base, y) - rmse(c, y):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in parts(base, y) - parts(c, y)) + ')')
np.savez(_repo_io('ridge-qm-strict.npz'), ridge=r1, logridge=l1)
