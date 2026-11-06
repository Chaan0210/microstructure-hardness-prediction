"""Harness: TabPFN-3.5 with an explicit n_estimators on a feature npz. usage: eval_v35n.py FEAT_NPZ N_EST SEED"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from harness import *
fnpz, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
F = np.load(_repo_io(fnpz), allow_pickle=True); X = F['X'].astype(float); ids = [str(i) for i in F['ids']]; y = labels(ids); nz = np.load(_repo_io('noise_train.npy'))
O = cv_predict(lambda: tabpfn_new('v3.5', seed, 'cpu', n), X, y, reps=(0, 1, 2)); tag = f'v35n{n}_{fnpz.replace(".npz","")}_s{seed}'; report(tag, O, y, nz); np.save(_repo_io(f'oof_{tag}.npy'), O)
