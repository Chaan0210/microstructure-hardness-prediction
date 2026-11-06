"""Append named columns from a reference feature npz (matched by id) to each H2 variant set. usage: addcols_ref.py REF_PREFIX COLS SUFFIX_IN SUFFIX_OUT"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
ref, cols, sin, sout = sys.argv[1], sys.argv[2].split(','), sys.argv[3], sys.argv[4]
for v in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4'):
    for s in ('train', 'test'):
        F = np.load(_repo_io(f'feat-{v}{sin}-{s}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]
        E = np.load(_repo_io(f'{ref}-{s}.npz'), allow_pickle=True); ek = [str(k) for k in E['keys']]; ep = {str(i): j for j, i in enumerate(E['ids'])}
        add = E['X'][[ep[i] for i in ids]][:, [ek.index(c) for c in cols]].astype(float)
        np.savez(_repo_io(f'feat-{v}{sout}-{s}.npz'), X=np.hstack([F['X'].astype(float), add]), keys=np.array([str(k) for k in F['keys']] + cols), ids=np.array(ids))
        print(v + sout, s, F['X'].shape[1] + len(cols), 'nan frac', np.isnan(add).mean(0).round(3))
