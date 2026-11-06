"""Append U-Net pixel-weighted block heterogeneity cols to the four 035 feature variants (same net mapping as their U-Net cols).
usage: make_het_sets.py COLS(comma) SUFFIX"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
cols = sys.argv[1].split(','); suf = sys.argv[2]
V = {'u1dm1sz': ['m1'], 'u1dm2sz': ['m2'], 'u1dmavg': ['m1', 'm2', 'h1'], 'u1dmavg4': ['m1', 'm2', 'h1', 'm3']}
for v, nets in V.items():
    for s in ('train', 'test'):
        F = np.load(_repo_io(f'../push-400/feat-{v}-{s}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]
        E = [np.load(_repo_io(f'uhet-{n}-{s}.npz'), allow_pickle=True) for n in nets]; ek = [str(k) for k in E[0]['keys']]
        for e in E: assert [str(i) for i in e['ids']] == ids
        add = np.mean([e['X'][:, [ek.index(c) for c in cols]] for e in E], 0)
        np.savez(_repo_io(f'feat-{v}{suf}-{s}.npz'), X=np.hstack([F['X'].astype(float), add]), keys=np.array([str(k) for k in F['keys']] + cols), ids=np.array(ids))
        print(v + suf, s, F['X'].shape[1] + len(cols))
