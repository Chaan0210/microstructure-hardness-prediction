"""Compare TabPFN CV when the noisy-half features come from different segmentation variants (paired folds)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np, torch; torch.set_num_threads(2)
from harness import *
Xv3, keys, ids = load_feats('features-v3'); Xtta, _, _ = load_feats('features-v3tta'); y = labels(ids); nz = np.load(_repo_io('noise_train.npy')); pos = {i: k for k, i in enumerate(ids)}
cache = {}
def run(name, X):
    O = cv_predict(lambda: tabpfn(8, 0, 'mps'), X, y, reps=(0, 1, 2)); report(name, O, y, nz); np.save(_repo_io(f'oof_var_{name}.npy'), O); return O
specs = sys.argv[1:]
for s in specs:
    if s == 'v3': run('v3', Xv3); continue
    var, mode = s.split(':')          # e.g. var-seg_down2-train-v1.npz:replace | :avg_tta
    V = np.load(_repo_io(var), allow_pickle=True); assert [str(k) for k in V['keys']] == keys
    rows = np.array([pos[str(i)] for i in V['ids']]); Xm = np.nanmean(V['X'], 1)
    if mode == 'replace': X = Xv3.copy(); X[rows] = Xm
    elif mode == 'avg_tta':  # average variant views with the existing 4-view TTA mean (weights by number of views)
        X = Xtta.copy(); nv = V['X'].shape[1]; X[rows] = (4 * Xtta[rows] + nv * Xm) / (4 + nv)
    elif mode == 'concat_noisy':  # extra columns: variant features for noisy rows, v3tta values elsewhere
        E = Xtta.copy(); E[rows] = Xm; X = np.hstack([Xtta, E])
    run(f'{var.replace(".npz", "")}_{mode}', X)
