"""Control: append products of K random column pairs (not chosen by any label/residual), seed-controlled.
usage: addctrl.py VARIANT SEED [K=2] -> feat-VARIANT-ctrlSEED-train.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
v, seed = sys.argv[1], int(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
F = np.load(_repo_io(f'../push-500/feat-{v}-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float)
rng = np.random.RandomState(1000 + seed); ok = [j for j, k in enumerate(keys) if k not in ('noise',)]
new, names = [], []
for _ in range(K):
    a, b = rng.choice(ok, 2, replace=False); new.append(X[:, a] * X[:, b]); names.append(f'{keys[a]}_x_{keys[b]}')
print(names)
np.savez(_repo_io(f'feat-{v}-ctrl{seed}-train.npz'), X=np.c_[X, np.array(new).T], keys=np.array(keys + names), ids=F['ids'])
