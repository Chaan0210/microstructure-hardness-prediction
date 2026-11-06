"""Append U-Net het-family cols (averaged over each variant's nets) to existing variant sets. usage: add_het.py COLS SUFFIX_IN SUFFIX_OUT"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
cols, sin, sout = sys.argv[1].split(','), sys.argv[2], sys.argv[3]
V = {'u1dm1sz': ['m1'], 'u1dm2sz': ['m2'], 'u1dmavg': ['m1', 'm2', 'h1'], 'u1dmavg4': ['m1', 'm2', 'h1', 'm3']}
for v, nets in V.items():
    for s in ('train', 'test'):
        F = np.load(_repo_io(f'feat-{v}{sin}-{s}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; fk = [str(k) for k in F['keys']]
        E = [np.load(_repo_io(f'uhet-{n}-{s}.npz'), allow_pickle=True) for n in nets]; ek = [str(k) for k in E[0]['keys']]
        for e in E: assert [str(i) for i in e['ids']] == ids
        chk = np.mean([e['X'][:, ek.index('het4_u')] for e in E], 0); assert np.allclose(chk, F['X'][:, fk.index('het4_u')]), 'het4_u mismatch'
        add = np.mean([e['X'][:, [ek.index(c) for c in cols]] for e in E], 0)
        np.savez(_repo_io(f'feat-{v}{sout}-{s}.npz'), X=np.hstack([F['X'].astype(float), add]), keys=np.array(fk + cols), ids=np.array(ids)); print(v + sout, s, len(fk) + len(cols))
