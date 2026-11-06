"""Independent check on the 100 train rows outside the Study900 dev-400 (never used for dev selection): fit on the 400 dev rows
(+ degraded copies of clean train images whose source is NOT a holdout row), predict the 100 holdout rows; compare noisy-row RMSE of the
plain vs sim-augmented RidgeCV / log-Ridge (production settings, 98 cols)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
dev = set(str(i) for i in np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['ids'])
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
NMIN = -1.0
def run(v, sets, model):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; X = F['X'].astype(float); y = np.array([lab[i] for i in ids])
    tr = np.array([i in dev for i in ids]); ho = ~tr; Xt, yt = X[tr], y[tr]
    for s in sets:
        S = np.load(_repo_io(f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); sk = [str(k) for k in S['keys']]
        keep = np.array([str(i) in dev for i in S['ids']]) & (S['X'][:, sk.index('noise')] * 255 >= NMIN)
        Xt = np.vstack([Xt, S['X'][keep].astype(float)]); yt = np.r_[yt, [lab[str(i)] for i in np.array(S['ids'])[keep]]]
    if model == 'ridge': p = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(Xt, yt).predict(X[ho])
    else: p = np.exp(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(Xt, np.log(yt)).predict(X[ho]))
    return p, y[ho], X[ho][:, [str(k) for k in F['keys']].index('noise')] * 255
for sets, NMIN in (([], -1.0), (['deg2', 'deg3'], -1.0), (['deg', 'deg2', 'deg3', 'deg4'], 12.0)):
    pr, y, nz = run('u1dmavgH2Z', sets, 'ridge'); pl = np.exp(np.mean([np.log(run(v, sets, 'log')[0]) for v in V], 0)); m = (pr + pl) / 2
    g = np.nan_to_num(nz, nan=0) >= 12
    print(f'+{"+".join(sets) or "none":10s} holdout n={len(y)} noisy n={g.sum()}: ridge noisy {np.sqrt(((pr - y)[g] ** 2).mean()):.3f}  logridge noisy {np.sqrt(((pl - y)[g] ** 2).mean()):.3f}  mean noisy {np.sqrt(((m - y)[g] ** 2).mean()):.3f} | clean+mid mean {np.sqrt(((m - y)[~g] ** 2).mean()):.3f}')
