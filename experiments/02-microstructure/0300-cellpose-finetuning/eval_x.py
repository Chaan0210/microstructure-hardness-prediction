"""Harness CV (500 imgs, 5-fold x 3 reps, same folds as oof_base_tabpfn.npy) for a feature matrix built from features-v3 plus row replacements.
usage: eval_x.py MODEL DEVICE SPEC [SPEC ...]
  MODEL: v2 (n_est 8, seed 0 = harness baseline) | v3.5 (default)
  SPEC : npz[:mode], mode = replace (rows <- mean over views, default) ; several SPECs are applied in order (e.g. noisy then clean npz)
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from harness import *
model, device = sys.argv[1], sys.argv[2]
X, keys, ids = load_feats('features-v3'); y = labels(ids); nz = np.load(_repo_io('noise_train.npy')); pos = {i: k for k, i in enumerate(ids)}
names = []
for s in sys.argv[3:]:
    f = s.split(':')[0]; V = np.load(_repo_io(f), allow_pickle=True); assert [str(k) for k in V['keys']] == keys
    rows = np.array([pos[str(i)] for i in V['ids']]); X[rows] = np.nanmean(V['X'], 1); names.append(f.replace('.npz', ''))
make = (lambda: tabpfn(8, 0, device)) if model == 'v2' else (lambda: tabpfn_new(model, 0, device))
O = cv_predict(make, X, y, reps=(0, 1, 2)); tag = f'{model}_' + '+'.join(names)
report(tag, O, y, nz); np.save(_repo_io(f'oof_x_{tag}.npy'), O)
report('base v2 v3tta (cached)', np.load(_repo_io('oof_base_tabpfn.npy')), y, nz)
