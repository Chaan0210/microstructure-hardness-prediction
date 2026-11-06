"""Harness with augmentation rows: degraded copies of held-out clean train images segmented by the ft model, labelled with the source
image's hardness, added ONLY when the source is in the training fold (no label leakage). usage: eval_aug2.py FEAT_NPZ AUG_NPZ[:view] [seed]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from harness import *
fnpz, aug = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
F = np.load(_repo_io(fnpz), allow_pickle=True); X = F['X'].astype(float); keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]; y = labels(ids); nz = np.load(_repo_io('noise_train.npy')); pos = {i: k for k, i in enumerate(ids)}
path, view = (aug.split(':') + ['1'])[:2]; A = np.load(_repo_io(path), allow_pickle=True); ak = [str(k) for k in A['keys']]
src = np.array([pos[str(i)] for i in A['ids']]); XA = np.full((len(src), X.shape[1]), np.nan)
cols = [keys.index(k) for k in ak]; XA[:, cols] = A['X'][:, int(view)]          # extra (vstd) columns stay NaN for augmentation rows
O = np.zeros((3, len(y)))
for ri, r in enumerate((0, 1, 2)):
    f = folds(len(y), r)
    for j in range(5):
        tr, va = f != j, f == j; ka = tr[src]
        m = tabpfn(8, seed, 'cpu'); m.fit(np.vstack([X[tr], XA[ka]]), np.r_[y[tr], y[src][ka]]); O[ri, va] = m.predict(X[va])
report(f'aug {path}:{view} on {fnpz}', O, y, nz)
