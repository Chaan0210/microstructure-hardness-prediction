"""Harness: ft features (noisy rows from ftseg npz, mean over views) + extra columns. mode zs: zero-shot v3tta copy of 26 key features;
mode vstd: across-view std of key features (noisy rows; NaN for clean rows)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from harness import *
model, device, ftnpz, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]; seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
X, keys, ids = load_feats('features-v3'); Xz, _, _ = load_feats('features-v3tta'); y = labels(ids); nz = np.load(_repo_io('noise_train.npy')); pos = {i: k for k, i in enumerate(ids)}
V = np.load(_repo_io(ftnpz), allow_pickle=True); rows = np.array([pos[str(i)] for i in V['ids']]); X[rows] = np.nanmean(V['X'], 1)
kk = ['sp_frac_grains', 'sp_count_frac', 'f_dark_20', 'f_dark_35', 'sizefield_range_s32', 'sizefield_std_s32', 'sizefield_std_s16', 'tile8_logA_std', 'grain_logA_std', 'grain_eqd_mean', 'aw_eqd_mean', 'n_obj', 'solidity_wmean', 'neighbors_std', 'ar_wmean', 'orient_coherence']
J = [keys.index(k) for k in kk]
if mode == 'zs': E = Xz[:, J]
else:
    E = np.full((len(y), len(J)), np.nan); E[rows] = np.nanstd(V['X'][:, :, J], 1)
Xu = np.hstack([X, E])
make = (lambda: tabpfn(8, seed, device)) if model == 'v2' else (lambda: tabpfn_new(model, seed, device))
O = cv_predict(make, Xu, y, reps=(0, 1, 2)); tag = f'concat_{model}_{mode}_{ftnpz.replace(".npz","")}' + (f'_s{seed}' if seed else ''); report(tag, O, y, nz); np.save(_repo_io(f'oof_{tag}.npy'), O)
