"""A-priori feature up-weighting by duplication (no label use): append copies of the U-Net physics columns.
usage: dup_cols.py BASE_PREFIX OUT_PREFIX"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
base, out = sys.argv[1:3]
COLS = ['u_ph', 'u_ph_hard', 'm_ph_hard', 'z_f25', 'z_f50', 'het4_u', 'het4_u_mx', 'm_la_mean', 'm_la_std', 'm_el_glob', 'm_el_pix', 'n_obj']
for split in ('train', 'test'):
    F = np.load(_repo_io(f'{base}-{split}.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; j = [keys.index(c) for c in COLS]
    np.savez(_repo_io(f'{out}-{split}.npz'), X=np.c_[F['X'], F['X'][:, j]], keys=np.array(keys + [c + '_dup' for c in COLS]), ids=F['ids'])
print('ok', out)
