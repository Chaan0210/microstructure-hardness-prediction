"""CV: train = real rows of train folds + aug rows whose SOURCE is in train folds; validate on real rows only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np, torch; torch.set_num_threads(3)
from harness import *
aug_file = sys.argv[1]; reps = (0, 1, 2); FEATS = sys.argv[2].split(','); DEV = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
y = None; nz = np.load(_repo_io('noise_train.npy'))
A = np.load(_repo_io(aug_file), allow_pickle=True); XA, src = A['X'], A['src']
for feat in FEATS:
    X, keys, ids = load_feats(feat); y = labels(ids); assert [str(k) for k in A['keys']] == keys
    O0 = np.zeros((len(reps), len(y))); O1 = np.zeros_like(O0)
    for ri, r in enumerate(reps):
        f = folds(len(y), r)
        for j in range(5):
            tr, va = f != j, f == j
            m = tabpfn(8, 0, DEV); m.fit(X[tr], y[tr]); O0[ri, va] = m.predict(X[va])
            am = tr[src]
            m = tabpfn(8, 0, DEV); m.fit(np.vstack([X[tr], XA[am]]), np.r_[y[tr], y[src[am]]]); O1[ri, va] = m.predict(X[va])
    report(f'{feat} real only', O0, y, nz); report(f'{feat} real + aug', O1, y, nz)
    np.save(_repo_io(f'oof_{feat}_aug.npy'), O1); np.save(_repo_io(f'oof_{feat}_real.npy'), O0)
