"""Compare TabPFN v2 (cached OOF) with newer TabPFN checkpoints on the same folds/features."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np, torch; torch.set_num_threads(2)
from harness import *
X, keys, ids = load_feats(sys.argv[2] if len(sys.argv) > 2 else 'features-v3tta'); y = labels(ids); nz = np.load(_repo_io('noise_train.npy'))
report('v2 n8 (cached)', np.load(_repo_io('oof_base_tabpfn.npy')), y, nz)
ver = sys.argv[1]
O = cv_predict(lambda: tabpfn_new(ver, 0, 'mps'), X, y, reps=(0, 1, 2)); report(f'{ver} default', O, y, nz); np.save(_repo_io(f'oof_new_{ver}.npy'), O)
