"""Append z-channel U-Net columns (unet4-<NET>-{split}.npz) to each variant set. usage: add_z.py NET COLS SUFFIX_IN SUFFIX_OUT"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
net, cols, sin, sout = sys.argv[1], sys.argv[2].split(','), sys.argv[3], sys.argv[4]
for v in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4'):
    for s in ('train', 'test'):
        F = np.load(_repo_io(f'feat-{v}{sin}-{s}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]
        Z = np.load(_repo_io(f'../push-400/unet4-{net}-{s}.npz'), allow_pickle=True); zk = [str(k) for k in Z['keys']]; assert [str(i) for i in Z['ids']] == ids
        add = Z['X'][:, [zk.index(c) for c in cols]].astype(float)
        np.savez(_repo_io(f'feat-{v}{sout}-{s}.npz'), X=np.hstack([F['X'].astype(float), add]), keys=np.array([str(k) for k in F['keys']] + cols), ids=np.array(ids))
        print(v + sout, s, F['X'].shape[1] + len(cols))
