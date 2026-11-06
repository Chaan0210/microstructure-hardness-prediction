"""Append a-priori interaction columns to a feature set (label-free, computed from existing columns only).
usage: addint.py VARIANT SPEC   (VARIANT e.g. u1dmavgH2Z; SPEC e.g. hetel) -> feat-VARIANT-SPEC-{train,test}.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
v, spec = sys.argv[1], sys.argv[2]
SPECS = {
    'hetel': [('het4_u', 'm_el_pix'), ('sizefield_std_s32', 'm_el_pix')],
}
for split in ('train', 'test'):
    F = np.load(_repo_io(f'../push-500/feat-{v}-{split}.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float)
    new = []; names = []
    for a, b in SPECS[spec]:
        new.append(X[:, keys.index(a)] * X[:, keys.index(b)]); names.append(f'{a}_x_{b}')
    np.savez(_repo_io(f'feat-{v}-{spec}-{split}.npz'), X=np.c_[X, np.array(new).T], keys=np.array(keys + names), ids=F['ids'])
print('done', v, spec)
