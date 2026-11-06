"""13 geometry columns per image: exact zero-shot-mask values for clean images (pixel noise < 9.5, geom-exact-<split>.npz), geometry-CNN
estimates (geom-cnn<tag>-<split>.npz) for every other image (never used to train the CNN). Appended to the 044-slot feature sets
(push-800/feat-<v>-anghcor-<split>.npz) -> feat-<v>-anghcorG<tag>-<split>.npz; also saves geomcols<tag>-<split>.npz.
usage: geom_feats.py TAG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]; tag = sys.argv[1]
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
for split in ('train', 'test'):
    C = np.load(_repo_path(H, f'geom-cnn{tag}-{split}.npz')); cid = [str(i) for i in C['ids']]; keys = [str(k) for k in C['keys']]
    X = C['X'].astype(float).copy(); Ex = np.load(_repo_path(H, f'geom-exact-{split}.npz')); epos = {str(i): k for k, i in enumerate(Ex['ids'])}
    src = np.array(['cnn'] * len(cid), dtype=object)
    for n, i in enumerate(cid):
        if i in epos: X[n] = Ex['X'][epos[i]]; src[n] = 'exact'
    np.savez(_repo_path(H, f'geomcols{tag}-{split}.npz'), ids=np.array(cid), X=X, keys=np.array(keys), source=src.astype(str))
    pos = {i: k for k, i in enumerate(cid)}
    for v in V:
        F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-{split}.npz'), allow_pickle=True); fid = [str(i) for i in F['ids']]
        Xa = np.c_[F['X'].astype(float), X[[pos[i] for i in fid]]]
        np.savez(_repo_path(H, f'feat-{v}-anghcorG{tag}-{split}.npz'), ids=F['ids'], X=Xa, keys=np.array([str(k) for k in F['keys']] + ['g_' + k for k in keys]))
    print(split, 'exact rows', int((src == 'exact').sum()), 'cnn rows', int((src == 'cnn').sum()))
