"""Exact per-source total-weight control for the DTA partners (gated rows: RidgeCV u1dmavgH2Z, logRidge x 4 variants; copies deg..deg4, noise >= 12,
copy set as production: every copy whose source is not in the outer validation fold).
  A  production (unweighted; RidgeCV GCV over [0.1..1000] / logRidge alpha 2)
  B  unweighted, alpha chosen by inner 5-fold CV GROUPED BY SOURCE (a source's real row and all its copies in the same inner fold),
     score = MSE on the real inner-validation rows with noise >= 12 (target domain)
  C  per-source equal total weight: the source's real row (if in training) and each of its c copies get 1/(1+c) (copies of a source whose real row
     is not in training get 1/c); weights rescaled so sum(w) = number of rows (keeps the Ridge penalty scale); alpha chosen as in B (weighted fits).
Outputs dev OOF [3,400] for the gated rows' partners -> srcw.npz; and holdout-100 predictions (fit on dev-400 + copies of dev sources) -> srcw-ho.npz."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import GroupKFold
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); sid = load(3)['ids']; dev = set(sid); lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; GR = {'ridge': [0.1, 0.3, 1, 3, 10, 30, 100, 1000], 'log': [0.3, 1, 2, 3, 10, 30, 100]}
def data(v):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise'); SX, SID = [], []
    for s in ['deg', 'deg2', 'deg3', 'deg4']:
        S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
    XA = F['X'].astype(float); return {str(i): k for k, i in enumerate(F['ids'])}, XA, np.nan_to_num(XA[:, ni] * 255, nan=0), np.vstack(SX), np.array(SID)
def pipe(kind, a): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True) if kind == 'ridge' else SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=a))
def tf(kind, y): return np.log(y) if kind == 'log' else y
def itf(kind, p): return np.exp(p) if kind == 'log' else p
def weights(src_real, src_copy):
    """per-source equal total weight, rescaled to sum = n rows"""
    from collections import Counter
    c = Counter(src_copy); real = set(src_real)
    wr = np.array([1.0 / (1 + c.get(s, 0)) for s in src_real]); wc = np.array([1.0 / (c[s] + (1 if s in real else 0)) for s in src_copy])
    w = np.r_[wr, wc]; return w * len(w) / w.sum()
def fit_arm(arm, kind, Xr, yr, sr, nzr, Xc, yc, sc):
    X = np.vstack([Xr, Xc]); y = np.r_[yr, yc]; src = np.r_[sr, sc]; isreal = np.r_[np.ones(len(yr), bool), np.zeros(len(yc), bool)]; nzall = np.r_[nzr, np.full(len(yc), 99.0)]
    if arm == 'A':
        if kind == 'ridge': m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(X, y)
        else: m = pipe(kind, 2.0).fit(X, tf(kind, y))
        return m, None
    W = weights(sr, sc) if arm == 'C' else np.ones(len(y)); us = np.unique(src); best = None
    for a in GR[kind]:
        err = []
        for tr, va in GroupKFold(5).split(X, y, src):
            wt = weights(src[tr][isreal[tr]], src[tr][~isreal[tr]]) if arm == 'C' else np.ones(tr.size)
            # weights() orders real rows first then copies: reorder to match tr order
            if arm == 'C':
                order = np.r_[np.where(isreal[tr])[0], np.where(~isreal[tr])[0]]; wtt = np.empty(tr.size); wtt[order] = wt; wt = wtt
            m = pipe(kind, a).fit(X[tr], tf(kind, y[tr]), ridge__sample_weight=wt); sel = va[isreal[va] & (nzall[va] >= 12)]
            if sel.size: err += list((itf(kind, m.predict(X[sel])) - y[sel]) ** 2)
        e = np.mean(err)
        if best is None or e < best[0]: best = (e, a)
    return pipe(kind, best[1]).fit(X, tf(kind, y), ridge__sample_weight=W), best[1]
D = {v: data(v) for v in V}; out = {}; ho = {}; alphas = {}
for arm in ('A', 'B', 'C'):
    for kind, vs in (('ridge', ['u1dmavgH2Z']), ('log', V)):
        O = []; H = []; al = []
        for v in vs:
            pos, XA, NZ, SX, SID = D[v]; o = np.zeros((3, 400))
            for p_, path in enumerate(splits):
                d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
                for k in range(5):
                    tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vset = set(va); kp = np.array([s not in vset for s in SID])
                    jt = np.array([pos[i] for i in tr]); m, a = fit_arm(arm, kind, XA[jt], np.array([lab[i] for i in tr]), np.array(tr), NZ[jt], SX[kp], np.array([lab[s] for s in SID[kp]]), SID[kp])
                    al.append(a); o[p_, [sid.index(i) for i in va]] = itf(kind, m.predict(XA[[pos[i] for i in va]]))
            O.append(o)
            ids_all = list(pos); trd = [i for i in ids_all if i in dev]; hd = [i for i in ids_all if i not in dev]; kp = np.array([s in dev for s in SID]); jt = np.array([pos[i] for i in trd])
            m, a = fit_arm(arm, kind, XA[jt], np.array([lab[i] for i in trd]), np.array(trd), NZ[jt], SX[kp], np.array([lab[s] for s in SID[kp]]), SID[kp]); H.append(itf(kind, m.predict(XA[[pos[i] for i in hd]]))); ho['hid'] = np.array(hd)
        key = 'ridge' if kind == 'ridge' else 'logridge'
        out[f'{arm}_{key}'] = O[0] if kind == 'ridge' else np.exp(np.mean(np.log(O), 0)); ho[f'{arm}_{key}'] = H[0] if kind == 'ridge' else np.exp(np.mean(np.log(H), 0))
        alphas[f'{arm}_{key}'] = al; print(arm, key, 'alpha chosen (median, min, max):', None if al[0] is None else (np.median(al), min(al), max(al)), flush=True)
np.savez(_repo_io('srcw.npz'), **out); np.savez(_repo_io('srcw-ho.npz'), **ho)
