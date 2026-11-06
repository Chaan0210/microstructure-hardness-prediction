"""Build feature sets: VARIANT + extra columns. spec 'angh' = ang_c4, ang_s4 (angle-SPLIT.npz) + het4_u*m_el_pix, sizefield_std_s32*m_el_pix.
usage: build_sets.py SPEC VARIANT [VARIANT...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
spec = sys.argv[1]
for v in sys.argv[2:]:
    for split in ('train', 'test'):
        F = np.load(_repo_io(f'../push-500/feat-{v}-{split}.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float)
        A = np.load(_repo_io(f'angle-{split}.npz'), allow_pickle=True); assert [str(i) for i in A['ids']] == [str(i) for i in F['ids']]
        new, names = [], []
        if 'ang' in spec or spec == 'angh':
            new += [A['X'][:, 0], A['X'][:, 1]]; names += ['ang_c4', 'ang_s4']
        if spec == 'angh':
            for a in ('het4_u', 'sizefield_std_s32'):
                new.append(X[:, keys.index(a)] * X[:, keys.index('m_el_pix')]); names.append(f'{a}_x_m_el_pix')
        np.savez(_repo_io(f'feat-{v}-{spec}-{split}.npz'), X=np.c_[X, np.array(new).T], keys=np.array(keys + names), ids=F['ids'])
    print('built', v, spec, names)
