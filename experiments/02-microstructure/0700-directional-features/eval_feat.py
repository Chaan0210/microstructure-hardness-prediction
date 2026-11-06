"""Harness: TabPFN v2 n8 (or v3.5) on a feature npz, seeds given; compares to cached 030 baseline with the same seed. usage: eval_feat.py NPZ MODEL SEEDS..."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-300'))
import harness
fn, model = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in sys.argv[3:]]
F = np.load(_repo_io(fn), allow_pickle=True); X = F['X'].astype(float); ids = [str(i) for i in F['ids']]; y = harness.labels(ids); nz = np.load(_repo_io('../push-300/noise_train.npy'))
Os = []; Bs = []
for s in seeds:
    make = (lambda: harness.tabpfn(8, s, __import__('os').environ.get('DEV','mps'))) if model == 'v2' else (lambda: harness.tabpfn_new(model, s, __import__('os').environ.get('DEV','mps')))
    O = harness.cv_predict(make, X, y, reps=(0, 1, 2)); Os.append(O); np.save(_repo_io(f'oof_{model}_{fn.replace(".npz","")}_s{s}.npy'), O)
    harness.report(f'{fn} {model} s{s}', O, y, nz)
    bf = f'../push-300/oof_none_v2_feat-a1s250n4vstd-stcoh-train_s{s}.npy' if model == 'v2' else None
    if bf: B = np.load(_repo_io(bf)); Bs.append(B); harness.report(f'  base 030 v2 s{s}', B, y, nz)
if len(Os) > 1:
    harness.report(f'{fn} {model} seed-mean', np.mean(Os, 0), y, nz)
    if Bs: harness.report('  base 030 seed-mean', np.mean(Bs, 0), y, nz)
