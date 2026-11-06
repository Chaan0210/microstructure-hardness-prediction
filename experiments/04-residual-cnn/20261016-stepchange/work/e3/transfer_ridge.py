"""E3 (frozen 2026-10-16 22:20, before any result): transfer ridge for gated rows (noise >= 12 & not failed).
Pooled partners assume one coefficient vector for clean and noisy rows (noise only shifts the intercept); under measurement attenuation the noisy
domain needs its own coefficients, but 100 noisy training rows cannot estimate 98 freely. Transfer ridge: beta_noisy = argmin ||y_n - b0 - X_n beta||^2
+ lam ||beta - beta_clean||^2, i.e. shrink toward the clean-domain coefficients instead of toward zero.
Data: push-800 feat-u1dmavgH2Z-anghcor (98 cols, the production Ridge partner set), median-impute + standardize on training-fold real rows.
beta_clean: RidgeCV(alphas .1,1,10,100,1000) on training-fold rows with noise < 12. lam: inner 5-fold CV on training-fold noisy rows, grid
{.3,1,3,10,30,100,300,1000,3000}. Arms: TR (real noisy rows only) and TRD (real noisy rows + degraded copies with noise >= 12 of training sources).
Outer: Study900 dev 400 x official 3 partitions x 5 folds. Decision in e3_eval.py: same-weight replacement of the production DTA RidgeCV partner
(0.128) on gated rows; promote only if blend gain >= 0.02, P >= 0.9, all partitions positive; else reject.
usage: transfer_ridge.py -> work/e3/oof-tr.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[4]; ROOT = _RepoPath(__file__).resolve().parents[5]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(ROOT, 'data/train.csv')))}
F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; XA = F['X'].astype(float); pos = {str(i): k for k, i in enumerate(F['ids'])}
ni = keys.index('noise')
SX, SID = [], []
for s in ['deg', 'deg2', 'deg3', 'deg4']:
    S = np.load(_repo_path(E, f'push-1200/synthrows-{s}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
SX = np.vstack(SX); SID = np.array(SID)
LAMS = [.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
def tr_fit(Xn, yn, bc, lam):   # centred closed form: (Xc'Xc + lam I) d = Xc'(yc - Xc bc); beta = bc + d; intercept from means
    mx, my = Xn.mean(0), yn.mean(); Xc = Xn - mx; r = (yn - my) - Xc @ bc
    d = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ r); b = bc + d; return b, my - mx @ b
def fit_predict(X, y, nzv, Xv, copies=None):
    med = np.nanmedian(X, 0); med = np.where(np.isnan(med), 0, med); f = lambda a: np.where(np.isnan(a), med, a)
    Xi = f(X); mu, sd = Xi.mean(0), Xi.std(0); sd[sd == 0] = 1; z = lambda a: (f(a) - mu) / sd
    Z = z(X); cl = nzv < 12; no = ~cl
    rc = RidgeCV(alphas=[.1, 1, 10, 100, 1000]).fit(Z[cl], y[cl]); bc = rc.coef_
    Zn, yn = Z[no], y[no]
    if copies is not None: Zn = np.vstack([Zn, z(copies[0])]); yn = np.r_[yn, copies[1]]
    nreal = no.sum(); err = {l: [] for l in LAMS}
    for tri, vai in KFold(5, shuffle=True, random_state=0).split(np.arange(nreal)):
        keep = np.r_[tri, np.arange(nreal, len(yn))]   # copies always in the inner training part; inner validation = real noisy rows only
        for l in LAMS:
            b, b0 = tr_fit(Zn[keep], yn[keep], bc, l); err[l] += list((Zn[vai] @ b + b0 - yn[vai]) ** 2)
    lam = min(LAMS, key=lambda l: np.mean(err[l])); b, b0 = tr_fit(Zn, yn, bc, lam)
    return z(Xv) @ b + b0, lam
splits = {'original': _repo_path(E, 'resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(E, 'resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(E, 'resolution-repartition-217/20261101/ensemble-OOF.npz')}
save = {}; lams = {'tr': [], 'trd': []}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; nzv = np.nan_to_num(X[:, ni] * 255, nan=0)
    p_tr, p_trd = np.zeros(len(y)), np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k
        p_tr[va], l1 = fit_predict(X[tr], y[tr], nzv[tr], X[va]); lams['tr'].append(l1)
        trs = set(np.array(ids)[tr]); keep = np.array([s in trs for s in SID])
        p_trd[va], l2 = fit_predict(X[tr], y[tr], nzv[tr], X[va], (SX[keep], np.array([lab[s] for s in SID[keep]]))); lams['trd'].append(l2)
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_tr': p_tr, f'{name}_trd': p_trd})
    print(name, 'done', flush=True)
np.savez(_repo_path(HERE, 'oof-tr.npz'), **save, lam_tr=np.array(lams['tr']), lam_trd=np.array(lams['trd'])); print('lambdas tr', lams['tr'], '\nlambdas trd', lams['trd'])
