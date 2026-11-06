from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np, torch; torch.set_num_threads(2)
from harness import *
X, keys, ids = load_feats('features-v3tta'); y = labels(ids); nz = np.load(_repo_io('noise_train.npy'))
R = np.load(_repo_io('robust-train.npz'), allow_pickle=True); RX = R['X']; Rk = [str(k) for k in R['keys']]
ROB_CP = ['sp_frac_grains', 'sp_count_frac', 'f_dark_20', 'f_dark_35', 'z_q75', 'grain_eqd_mean', 'aw_eqd_mean', 'ar_wmean', 'orient_coherence', 'px_pore_n', 'px_pore_frac', 'noise']
cp = X[:, [keys.index(k) for k in ROB_CP]]
sets = {'robust20': RX, 'robust20+cp12': np.hstack([RX, cp])}
for name in sys.argv[1].split(','):
    O = cv_predict(lambda: tabpfn(8, 0, 'mps'), sets[name], y, reps=(0, 1, 2)); report(name, O, y, nz); np.save(_repo_io(f'oof_set_{name}.npy'), O)
