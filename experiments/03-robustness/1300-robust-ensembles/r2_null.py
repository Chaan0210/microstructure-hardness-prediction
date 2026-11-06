"""Second-round group screen for the plain partners on non-gated rows, on top of the nb column set (073): drop each remaining group vs random drops
of the same size from the nb set (100 draws). Metric: non-gated blend dMSE vs the 073 dev blend (RidgeCV + logRidge u1dmavgH2Z deltas)."""
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
P = Path(_repo_io('../..')); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
Y = np.array([lab[str(i)] for i in F['ids']]); idx = []
for p in splits:
    d = np.load(_repo_io(p)); ids, fold = [str(i) for i in d['ids']], d['fold']
    for k in range(5): idx.append((len(idx) // 5, np.array([pos[i] for i, f in zip(ids, fold) if f != k]), np.array([pos[i] for i, f in zip(ids, fold) if f == k]), [sid.index(i) for i, f in zip(ids, fold) if f == k]))
def oof(cols, kind):
    O = np.zeros((3, 400))
    for p_, tr, va, vi in idx:
        m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
        m.fit(XA[tr][:, cols], np.log(Y[tr]) if kind == 'log' else Y[tr]); pr = m.predict(XA[va][:, cols]); O[p_, vi] = np.exp(pr) if kind == 'log' else pr
    return O
G = json.load(open(_repo_io('abl/groups.json'))); nbdrop = set(G['intensity'].split() + G['het'].split()); nbcols = [j for j, k in enumerate(keys) if k not in nbdrop]
y, g, fail = N.y, N.g, N.fail; ng = ~g & ~fail[0]; base = np.load(_repo_io('dev-nb-candidate.npy')); R0, L0 = oof(nbcols, 'ridge'), oof(nbcols, 'log')
def gain(cols):
    R, L = oof(cols, 'ridge'), oof(cols, 'log'); new = base + np.where(ng, 0.128 * (R - R0) + 0.128 * L0 * (L / L0 - 1) * (1.0), 0)
    return ((new - y) ** 2 - (base - y) ** 2)[:, ng].mean()
cand = [j for j in nbcols if keys[j] not in ('noise', 'n_obj')]; rng = np.random.RandomState(0); nulls = {}
for gname, cs in G.items():
    if gname in ('het', 'intensity'): continue
    drop = set(cs.split()); n = len(drop); obs = gain([j for j in nbcols if keys[j] not in drop])
    if n not in nulls: nulls[n] = np.array([gain([j for j in nbcols if j not in set(rng.choice(cand, n, replace=False))]) for _ in range(100)])
    nl = nulls[n]; print(f'{gname:9s} ({n:2d}) non-gated dMSE vs 073 {obs:+.2f} | random null mean {nl.mean():+.2f} sd {nl.std():.2f} | P {(nl <= obs).mean():.2f}', flush=True)
