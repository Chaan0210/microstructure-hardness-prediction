"""Harness: noise-shift correction (noise_correct.py) on the ft-a1s250 feature set; keeps raw 'noise' column. Appends corrected columns or replaces."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from harness import *
import noise_correct as nc
model, device, mode, how = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
F = np.load(_repo_io('feat-a1s250n1-train.npz'), allow_pickle=True); X = F['X'].astype(float); keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
y = labels(ids); nz = np.load(_repo_io('noise_train.npy')); jn = keys.index('noise')
M = nc.fit(X, X[:, jn], skip={jn}); Xc = nc.apply(X, X[:, jn], M, mode)
Xu = Xc if how == 'replace' else np.hstack([X, Xc[:, [j for j in range(X.shape[1]) if j != jn]]])
make = (lambda: tabpfn(8, 0, device)) if model == 'v2' else (lambda: tabpfn_new(model, 0, device))
O = cv_predict(make, Xu, y, reps=(0, 1, 2)); tag = f'nc_{model}_{mode}_{how}'; report(tag, O, y, nz); np.save(_repo_io(f'oof_{tag}.npy'), O)
report('ft v2 (cached)', np.load(_repo_io('oof_var_ftseg-ft-a1s250-train-noisy-n1_replace.npy')), y, nz)
