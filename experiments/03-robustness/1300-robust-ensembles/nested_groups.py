"""[10-15 fix: inner selection also excludes copies of OUTER-validation sources] Honest (nested) feature-group selection for the linear partners. In each outer fold: inner 5-fold CV on the outer-training rows decides, by
backward elimination over the 13 groups (drop the group whose removal lowers inner CV MSE most, repeat while it improves), the column set for
(a) the plain RidgeCV partner (criterion: inner CV MSE on non-gated real rows) and (b) the DTA partner (criterion: gated real rows; copies of
inner-validation sources excluded). The same column sets are used for logRidge (4 variants). Outer-fold predictions -> 061 blend."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
Gr = json.load(open(_repo_io('abl/groups.json'))); V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
def data(v):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
    SX, SID = [], []
    for s in ['deg', 'deg2', 'deg3', 'deg4']:
        S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
    return keys, {str(i): k for k, i in enumerate(F['ids'])}, F['X'].astype(float), np.vstack(SX), np.array(SID)
D = {v: data(v) for v in V}; keys = D['u1dmavgH2Z'][0]; ni = keys.index('noise')
gcols = {g: [keys.index(c) for c in cs.split()] for g, cs in Gr.items()}
def mk(kind): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
def fitpred(kind, Xt, yt, Xv): m = mk(kind).fit(Xt, np.log(yt) if kind == 'log' else yt); p = m.predict(Xv); return np.exp(p) if kind == 'log' else p
def cols_of(drop): dd = set(c for g in drop for c in gcols[g]); return [j for j in range(len(keys)) if j not in dd]
def inner_mse(tr, drop, dta, gate_of, vout=frozenset()):
    keys_, pos, XA, SX, SID = D['u1dmavgH2Z']; cols = cols_of(drop); tr = np.array(tr); err = []
    for a, b in KFold(5, shuffle=True, random_state=0).split(tr):
        ta, tb = tr[a], tr[b]; Xt = XA[[pos[i] for i in ta]][:, cols]; yt = np.array([lab[i] for i in ta])
        if dta: bs = set(tb); kp = np.array([(s not in bs) and (s not in vout) for s in SID]); Xt = np.vstack([Xt, SX[kp][:, cols]]); yt = np.r_[yt, [lab[s] for s in SID[kp]]]
        sel = np.array([gate_of[i] == dta for i in tb])
        if sel.sum() == 0: continue
        pv = fitpred('ridge', Xt, yt, XA[[pos[i] for i in tb[sel]]][:, cols]); err += list((pv - np.array([lab[i] for i in tb[sel]])) ** 2)
    return np.mean(err)
def select(tr, dta, gate_of, vout):
    drop = []; best = inner_mse(tr, drop, dta, gate_of, vout)
    while True:
        cand = [(inner_mse(tr, drop + [g], dta, gate_of, vout), g) for g in Gr if g not in drop]
        if not cand: break
        m, g = min(cand)
        if m < best: best, drop = m, drop + [g]
        else: break
    return drop
nzall = {i: D['u1dmavgH2Z'][2][k, ni] * 255 for i, k in D['u1dmavgH2Z'][1].items()}
F0 = np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)
out = {f'{k}_{d}': np.zeros((3, 400)) for k in ('ridge', 'logridge') for d in (0, 1)}; log = []
for p_, path in enumerate(splits.values()):
    dd = np.load(_repo_io(path)); ids, fold = [str(i) for i in dd['ids']], dd['fold']
    for k in range(5):
        tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va)
        gate_of = {i: int(np.nan_to_num(nzall[i], nan=0) >= 12) for i in tr}
        for dta in (0, 1):
            drop = select(tr, dta, gate_of, frozenset(vs)); log.append((p_, k, dta, drop)); cols = cols_of(drop)
            for kind, vs_ in (('ridge', ['u1dmavgH2Z']), ('log', V)):
                preds = []
                for v in vs_:
                    _, pos, XA, SX, SID = D[v]; Xt = XA[[pos[i] for i in tr]][:, cols]; yt = np.array([lab[i] for i in tr])
                    if dta: kp = np.array([s not in vs for s in SID]); Xt = np.vstack([Xt, SX[kp][:, cols]]); yt = np.r_[yt, [lab[s] for s in SID[kp]]]
                    preds.append(fitpred(kind, Xt, yt, XA[[pos[i] for i in va]][:, cols]))
                pr = preds[0] if kind == 'ridge' else np.exp(np.mean(np.log(preds), 0))
                out[f'{"ridge" if kind == "ridge" else "logridge"}_{dta}'][p_, [sid.index(i) for i in va]] = pr
            print(p_, k, 'dta' if dta else 'plain', 'drop', drop, flush=True)
np.savez(_repo_io('rr-nested2.npz'), **out); json.dump(log, open(_repo_io('rr-nested2.json'), 'w'))
