"""usage: addcols600.py BASE_PREFIX EXTRA_PREFIX COLS(comma|all) OUT_PREFIX  -> OUT-{train,test}.npz = base + selected extra columns (matched by id)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, os
base, extra, cols, out = sys.argv[1:5]
for split in ('train', 'test'):
    if not os.path.exists(f'{base}-{split}.npz') or not os.path.exists(f'{extra}-{split}.npz'): print('skip', split); continue
    B = np.load(_repo_io(f'{base}-{split}.npz'), allow_pickle=True); E = np.load(_repo_io(f'{extra}-{split}.npz'), allow_pickle=True)
    ek = [str(k) for k in E['keys']]; sel = ek if cols == 'all' else cols.split(','); pos = {str(i): k for k, i in enumerate(E['ids'])}
    Xe = E['X'][[pos[str(i)] for i in B['ids']]][:, [ek.index(c) for c in sel]]
    np.savez(_repo_io(f'{out}-{split}.npz'), X=np.c_[B['X'], Xe], keys=np.array([str(k) for k in B['keys']] + sel), ids=B['ids']); print(out, split, np.c_[B['X'], Xe].shape)
