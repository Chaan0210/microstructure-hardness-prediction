"""Pessimistic version of make_degph.py: every column except 'noise' is degraded on clean rows to correlation rho (shared per-image error 70% +
independent 30%), i.e. the whole measurement is as unreliable as rho. usage: make_degall.py RHO TAG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; rho, tag = float(sys.argv[1]), sys.argv[2]
F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); K = [str(k) for k in F['keys']]; X = F['X'].astype(float).copy()
cl = np.nan_to_num(X[:, K.index('noise')] * 255, nan=99) < 9.5; r = np.random.RandomState(4343); e = r.normal(size=cl.sum())
for j, c in enumerate(K):
    if c == 'noise': continue
    x = X[cl, j]
    if np.all(np.isnan(x)): continue
    mu, sd = np.nanmean(x), np.nanstd(x); ee = np.sqrt(0.7) * e * np.sign(np.nan_to_num(np.corrcoef(np.nan_to_num(x, nan=mu), e)[0, 1]) + 1e-9) + np.sqrt(0.3) * r.normal(size=cl.sum())
    X[cl, j] = mu + rho * (x - mu) + np.sqrt(1 - rho ** 2) * sd * ee
np.savez(_repo_path(H, f'feat-u1dmavgH2Z-anghcor{tag}-train.npz'), ids=F['ids'], X=X, keys=F['keys']); print(tag, 'done')
