"""Gated (noise >= 12) DTA partners: group drops vs random-column-drop null. RidgeCV + logRidge (u1dmavgH2Z, logRidge delta as ratio on the production
4-variant partner), d1234 copies, Study900 folds; metric = noisy-row blend dMSE (061 structure). Null: 200 random drops of the same size as each group."""
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
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
import nsgain as N
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
v = 'u1dmavgH2Z'; F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float); Y = np.array([lab[str(i)] for i in F['ids']]); SX, SID = [], []
for s in ['deg', 'deg2', 'deg3', 'deg4']:
    S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
SX = np.vstack(SX); SID = np.array(SID); SY = np.array([lab[s] for s in SID])
idx = []
for p in splits:
    d = np.load(_repo_io(p)); ids, fold = [str(i) for i in d['ids']], d['fold']
    for k in range(5):
        va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va)
        idx.append((len(idx) // 5, np.array([pos[i] for i, f in zip(ids, fold) if f != k]), np.array([pos[i] for i in va]), [sid.index(i) for i in va], np.array([s not in vs for s in SID])))
def oof(cols, kind):
    O = np.zeros((3, 400))
    for p_, tr, va, vi, kp in idx:
        X = np.vstack([XA[tr][:, cols], SX[kp][:, cols]]); yy = np.r_[Y[tr], SY[kp]]
        m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
        m.fit(X, np.log(yy) if kind == 'log' else yy); pr = m.predict(XA[va][:, cols]); O[p_, vi] = np.exp(pr) if kind == 'log' else pr
    return O
y, g, c61, C = N.y, N.g, N.c61, N.C; allc = list(range(len(keys))); R0, L0 = oof(allc, 'ridge'), oof(allc, 'log')
def gain(cols):
    R, L = oof(cols, 'ridge'), oof(cols, 'log'); Cx = dict(C); Cx['ridge'] = np.where(g, C['ridge'] + R - R0, C['ridge']); Cx['logr'] = np.where(g, C['logr'] * L / L0, C['logr'])
    return ((N.blend(Cx) - y) ** 2 - (c61 - y) ** 2)[:, g].mean()
G = json.load(open(_repo_io('abl/groups.json'))); cand = [j for j, k in enumerate(keys) if k not in ('noise', 'n_obj')]; rng = np.random.RandomState(0); nulls = {}
for gname, cs in G.items():
    drop = set(cs.split()); n = len(drop); obs = gain([j for j, k in enumerate(keys) if k not in drop])
    if n not in nulls: nulls[n] = np.array([gain([j for j in allc if j not in set(rng.choice(cand, n, replace=False))]) for _ in range(100)])
    nl = nulls[n]; print(f'{gname:9s} ({n:2d} cols) noisy dMSE {obs:+.2f} | random-{n} null mean {nl.mean():+.2f} sd {nl.std():.2f} | P(null <= obs) {(nl <= obs).mean():.2f}', flush=True)
