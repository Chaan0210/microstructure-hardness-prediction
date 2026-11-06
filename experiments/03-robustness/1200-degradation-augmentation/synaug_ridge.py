"""Ridge / log-Ridge version of synaug_dev.py (fast): real training rows [+ degraded copies of clean train images, source not in the val fold]."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
P = Path(_repo_io('../..')); lab = {r.split(',')[0]: float(r.split(',')[1]) for r in open(_repo_path(P, 'data/train.csv')).read().split()[1:]}
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'),
          '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
def run(v, tags, model, wsyn=1.0):
    F = np.load(_repo_io(f'../push-500/feat-{v}-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
    SX, SY, SID = [], [], []
    for t in tags:
        S = np.load(_repo_io(f'../push-500/synthrows-train_{t}-{v}.npz'), allow_pickle=True); SX.append(S['X'].astype(float)); SID += [str(i) for i in S['ids']]; SY += [lab[str(i)] for i in S['ids']]
    SX = np.vstack(SX) if SX else np.zeros((0, len(keys))); SY = np.array(SY); SID = np.array(SID); E = {}
    for name, path in splits.items():
        d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; g = np.zeros(len(y)); nz = X[:, keys.index('noise')] * 255
        for k in range(5):
            tr, va = fold != k, fold == k; vids = set(np.array(ids)[va]); keep = np.array([s not in vids for s in SID], bool) if len(SID) else np.zeros(0, bool)
            Xt = np.vstack([X[tr], SX[keep]]); yt = np.r_[y[tr], SY[keep]]; w = np.r_[np.ones(tr.sum()), np.full(keep.sum(), wsyn)]
            if model == 'ridge': m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))); m.fit(Xt, yt, ridgecv__sample_weight=w); g[va] = m.predict(X[va])
            else: m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)); m.fit(Xt, np.log(yt), ridge__sample_weight=w); g[va] = np.exp(m.predict(X[va]))
        for b, msk in (('all', np.ones(len(y), bool)), ('clean', nz < 9.5), ('mid', (nz >= 9.5) & (nz < 12)), ('noisy', nz >= 12)): E.setdefault(b, []).append(((g - y)[msk] ** 2).mean())
    return {b: float(np.sqrt(np.mean(v))) for b, v in E.items()}
for model in ('ridge', 'logridge'):
    for v in ('u1dmavgH2Z', 'u1dm1szH2Z'):
        for tags in ([], ['deg'], ['deg2'], ['deg', 'deg2'], ['deg2', 'deg3']):
            r = run(v, tags, model); print(f'{model:8s} {v:11s} +{"+".join(tags) or "none":10s} ' + ' '.join(f'{b} {x:.3f}' for b, x in r.items()), flush=True)
