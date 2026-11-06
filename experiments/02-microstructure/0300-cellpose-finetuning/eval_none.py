"""Harness for a ready feature npz (no extra columns) with a TabPFN seed. usage: eval_none.py MODEL DEVICE FEAT_NPZ SEED"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from harness import *
model, device, fnpz, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
F = np.load(_repo_io(fnpz), allow_pickle=True); X = F['X'].astype(float); ids = [str(i) for i in F['ids']]; y = labels(ids); nz = np.load(_repo_io('noise_train.npy'))
make = (lambda: tabpfn(8, seed, device)) if model == 'v2' else (lambda: tabpfn_new(model, seed, device))
O = cv_predict(make, X, y, reps=(0, 1, 2)); tag = f'none_{model}_{fnpz.replace(".npz","")}_s{seed}'; report(tag, O, y, nz); np.save(_repo_io(f'oof_{tag}.npy'), O)
