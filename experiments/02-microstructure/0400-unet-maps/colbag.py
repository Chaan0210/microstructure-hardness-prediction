"""Harness: TabPFN-v2 n8 on random column subsets (feature bagging), averaged. usage: colbag.py FEAT_NPZ K FRAC [seed0]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-300'))
import harness
fn, K, frac = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); s0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
F = np.load(_repo_io(fn), allow_pickle=True); X = F['X'].astype(float); ids = [str(i) for i in F['ids']]; y = harness.labels(ids); nz = np.load(_repo_io('../push-300/noise_train.npy'))
rng = np.random.RandomState(1234 + s0); subs = [np.sort(rng.choice(X.shape[1], int(round(frac * X.shape[1])), replace=False)) for _ in range(K)]
O = np.zeros((3, len(y)))
for b, cols in enumerate(subs):
    Ob = harness.cv_predict(lambda: harness.tabpfn(8, b, 'mps'), X[:, cols], y, reps=(0, 1, 2)); O += Ob / K
    if b in (0, K - 1): harness.report(f'  bag member {b}', Ob, y, nz)
harness.report(f'colbag K={K} frac={frac}', O, y, nz); np.save(_repo_io(f'oof_colbag_{fn.replace(".npz","")}_K{K}_f{frac}.npy'), O)
