"""Harness: does training only on the same noise half help? For each fold, predict clean-half val rows with a model fit on (a) all train rows
(b) clean-half train rows only; noisy-half val rows with (a) all (b) noisy-half only. Features: feat-a1s250n1-train.npz."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from harness import *
F = np.load(_repo_io(sys.argv[1] if len(sys.argv) > 1 else 'feat-a1s250n1-train.npz'), allow_pickle=True); X = F['X'].astype(float); ids = [str(i) for i in F['ids']]; y = labels(ids); nz = np.load(_repo_io('noise_train.npy'))
half = nz > np.median(nz)
Oa = np.zeros((3, len(y))); Ob = np.zeros((3, len(y)))
for ri, r in enumerate((0, 1, 2)):
    f = folds(len(y), r)
    for j in range(5):
        tr, va = f != j, f == j
        m = tabpfn(8, 0, 'cpu'); m.fit(X[tr], y[tr]); Oa[ri, va] = m.predict(X[va])
        for h in (False, True):
            t2 = tr & (half == h); v2 = va & (half == h)
            m = tabpfn(8, 0, 'cpu'); m.fit(X[t2], y[t2]); Ob[ri, v2] = m.predict(X[v2])
report('all-rows model', Oa, y, nz); report('same-half model', Ob, y, nz); report('avg', (Oa + Ob) / 2, y, nz)
