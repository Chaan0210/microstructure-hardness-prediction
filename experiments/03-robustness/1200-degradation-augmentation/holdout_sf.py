"""Independent check (100 train rows outside Study900 dev-400) of the copy-realism fix: partners fitted on the 400 dev rows + copies of dev sources
(noise >= 12, sets deg..deg7, internal-CV alphas), with SHIFTFIX lam in {0, 0.5, 1} (shift estimated on the 400 dev rows), predict the holdout.
usage: holdout_sf.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
dev = set(str(i) for i in np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['ids'])
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; SETS = 'deg,deg2,deg3,deg4,deg5,deg6,deg7'.split(',')
A = [0.001, 0.01, 0.1, 1, 10, 100, 1000]; LA = [0.01, 0.1, 0.3, 1, 2, 3, 10, 30, 100]
def run(v, lam, model):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; X = F['X'].astype(float); y = np.array([lab[i] for i in ids])
    keys = [str(k) for k in F['keys']]; ni = keys.index('noise'); pos = {i: j for j, i in enumerate(ids)}; tr = np.array([i in dev for i in ids]); ho = ~tr
    CX, CY, CS = [], [], []
    for s in SETS:
        S = np.load(_repo_io(f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); k = np.array([str(i) in dev for i in S['ids']]) & (S['X'][:, ni] * 255 >= 12)
        CX.append(S['X'][k].astype(float)); CY += [lab[str(i)] for i in np.array(S['ids'])[k]]; CS += [pos[str(i)] for i in np.array(S['ids'])[k]]
    CX = np.vstack(CX)
    if lam > 0:
        nz = np.nan_to_num(X[:, ni] * 255, nan=0); real = np.nanmean(X[tr & (nz >= 12)], 0) - np.nanmean(X[tr & (nz < 9.5)], 0); sim = np.nanmean(CX - X[CS], 0)
        fix = np.nan_to_num(real - sim); fix[ni] = 0; CX = CX + lam * fix
    Xt, yt = np.vstack([X[tr], CX]), np.r_[y[tr], CY]
    if model == 'ridge': p = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=A)).fit(Xt, yt).predict(X[ho])
    else: p = np.exp(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=LA)).fit(Xt, np.log(yt)).predict(X[ho]))
    return p, y[ho], np.nan_to_num(X[ho][:, ni] * 255, nan=0)
import warnings; warnings.filterwarnings('ignore')
for lam in (0, 0.5, 1):
    pr, y, nz = run('u1dmavgH2Z', lam, 'ridge'); pl = np.exp(np.mean([np.log(run(v, lam, 'log')[0]) for v in V], 0)); g = nz >= 12; m = (pr + pl) / 2
    print(f'lam {lam}: holdout noisy n={g.sum()} ridge {np.sqrt(((pr - y)[g] ** 2).mean()):.3f} logridge {np.sqrt(((pl - y)[g] ** 2).mean()):.3f} mean {np.sqrt(((m - y)[g] ** 2).mean()):.3f} | mean-pred sd noisy {m[g].std():.2f} (y sd {y[g].std():.2f})')
