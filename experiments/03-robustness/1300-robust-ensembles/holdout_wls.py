"""Blend-level holdout-100 check of the WLS partner bundle (run from push-1200): partners and gmm-Ridge refit on dev-400 with inverse-variance
weights (variance model a + b/N + c*noisy from inner 5-fold OOF residuals of the unweighted fit on dev-400), proxy 061 structure of holdout_blend.py."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import KFold
src = open(_repo_io('holdout_blend.py')).read(); src = src[:src.index("e61 = (P['061'] - y) ** 2")]; exec(compile(src, 'hb', 'exec'))
def wls_run(v, sets, model):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); ids_ = [str(i) for i in F['ids']]; X = F['X'].astype(float); yy = np.array([lab[i] for i in ids_])
    keys = [str(k) for k in F['keys']]; ni, nn = keys.index('noise'), keys.index('n_obj'); trm = np.array([i in dev for i in ids_]); ho = ~trm
    SX, SY = [], []
    for s in sets:
        S = np.load(_repo_io(f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); k = np.array([str(i) in dev for i in S['ids']]) & (S['X'][:, ni] * 255 >= 12); SX.append(S['X'][k].astype(float)); SY += [lab[str(i)] for i in np.array(S['ids'])[k]]
    SX = np.vstack(SX) if SX else np.zeros((0, X.shape[1])); SY = np.array(SY)
    D = lambda Z: np.c_[np.ones(len(Z)), 1 / np.clip(np.nan_to_num(Z[:, nn], nan=20), 20, None), (np.nan_to_num(Z[:, ni] * 255, nan=0) >= 12).astype(float)]
    mk = lambda: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if model == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
    tf = (lambda t: np.log(t)) if model == 'log' else (lambda t: t); itf = (lambda t: np.exp(t)) if model == 'log' else (lambda t: t)
    def fit(Xa, ya, w): m = mk(); m.fit(Xa, tf(ya), **{m.steps[-1][0] + '__sample_weight': w}); return m
    Xr, yr = X[trm], yy[trm]; res = np.zeros(len(yr))
    for a, b in KFold(5, shuffle=True, random_state=0).split(Xr):   # copies of the inner-val sources are not excluded here (proxy; residual scale only)
        res[b] = yr[b] - itf(fit(np.vstack([Xr[a], SX]), np.r_[yr[a], SY], np.ones(len(a) + len(SY))).predict(Xr[b]))
    cf, _ = nnls(D(Xr), res ** 2); cf[0] = max(cf[0], 1.0); w = np.r_[1 / (D(Xr) @ cf), 1 / (D(SX) @ cf) if len(SX) else []]; w /= w.mean()
    return itf(fit(np.vstack([Xr, SX]), np.r_[yr, SY], w).predict(X[ho]))
rw0 = wls_run('u1dmavgH2Z', [], 'ridge'); lw0 = np.exp(np.mean([np.log(wls_run(v, [], 'log')) for v in V], 0))
s4 = 'deg,deg2,deg3,deg4'.split(','); rw1 = wls_run('u1dmavgH2Z', s4, 'ridge'); lw1 = np.exp(np.mean([np.log(wls_run(v, s4, 'log')) for v in V], 0))
# WLS gmm-Ridge on dev-400 -> holdout
Gw = np.load(_repo_io('../push-1300/wls-gmm.npz'))  # (dev OOF only; refit here)
Dg = np.c_[np.ones(len(XG)), 1 / np.clip(np.nan_to_num(F['X'][:, [str(k) for k in F['keys']].index('n_obj')].astype(float), nan=20), 20, None), (np.nan_to_num(F['X'][:, [str(k) for k in F['keys']].index('noise')].astype(float) * 255, nan=0) >= 12).astype(float)]
ti = np.where(tr)[0]; res = np.zeros(len(ti))
for a, b in KFold(5, shuffle=True, random_state=0).split(ti): res[b] = yy[ti[b]] - gp['fit_predict'](XG[ti[a]], yy[ti[a]], XG[ti[b]], 19001)[0]
cf, _ = nnls(Dg[ti], res ** 2); cf[0] = max(cf[0], 1.0); wg = 1 / (Dg[ti] @ cf); wg /= wg.mean()
from sklearn.model_selection import KFold as KF
sc = []
for a_ in gp['ALPHAS']:
    err = []
    for itr, iva in KF(3, shuffle=True, random_state=19001).split(ti):
        m = gp['ridge'](a_).fit(XG[ti[itr]], yy[ti[itr]], ridge__sample_weight=wg[itr]); err.append((wg[iva] * (m.predict(XG[ti[iva]]) - yy[ti[iva]]) ** 2).sum() / wg[iva].sum())
    sc.append(np.mean(err))
gmmw = gp['ridge'](gp['ALPHAS'][int(np.argmin(sc))]).fit(XG[ti], yy[ti], ridge__sample_weight=wg).predict(XG[~tr])
Pw = (0.16 * cc + 0.384 * sp + 0.128 * np.where(g, rw1, rw0) + 0.128 * np.where(g, lw1, lw0) + 0.15 * gmmw) / 0.95
e61 = (P['061'] - y) ** 2; ew = (Pw - y) ** 2; d = ew - e61
print(f'holdout: 061 {np.sqrt(e61.mean()):.3f} -> WLS bundle {np.sqrt(ew.mean()):.3f}; all dMSE {d.mean():+.2f} (SE {d.std() / np.sqrt(len(d)):.2f}); clean+mid dMSE {d[~g].mean():+.2f} noisy dMSE {d[g].mean():+.2f}')
print(f'components holdout RMSE: ridge {np.sqrt(((np.where(g, r61, r0) - y) ** 2).mean()):.3f} -> {np.sqrt(((np.where(g, rw1, rw0) - y) ** 2).mean()):.3f}; logr {np.sqrt(((np.where(g, l61, l0) - y) ** 2).mean()):.3f} -> {np.sqrt(((np.where(g, lw1, lw0) - y) ** 2).mean()):.3f}; gmm {np.sqrt(((gmm - y) ** 2).mean()):.3f} -> {np.sqrt(((gmmw - y) ** 2).mean()):.3f}')
