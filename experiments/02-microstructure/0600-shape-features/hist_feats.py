"""A-priori block of distributional readouts from U-Net maps (no residual screening). Label-free.
z-histogram (unet4-z1 darkness channel, grain-pixel weighted, 10 bins), phase-probability histogram (mean of unet2 m1,m2,h1,m3; 6 bins of p1/(p0+p1)),
log-area quantiles (5..95%, grain-pixel weighted, 7 values). usage: hist_feats.py SPLIT -> hist-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
split = sys.argv[1]; U = '../push-400'
M0 = [np.load(_repo_io(f'{U}/unet2maps-{n}-{split}.npz')) for n in ['m1', 'm2', 'h1', 'm3']]; ids = [str(i) for i in M0[0]['ids']]
for m in M0: assert [str(i) for i in m['ids']] == ids
P = np.mean([m['maps'][:, [0, 1, 4]].astype(np.float32) for m in M0], 0)
Z0 = np.load(_repo_io(f'{U}/unet4maps-z1-{split}.npz')); assert [str(i) for i in Z0['ids']] == ids; Z = Z0['maps'][:, [0, 1, 7]].astype(np.float32)
zb = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 10]; pb = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]; qs = [5, 20, 35, 50, 65, 80, 95]
keys = [f'zh{k}' for k in range(10)] + [f'ph{k}' for k in range(6)] + [f'laq{q}' for q in qs]; X = []
def wq(v, w, q):
    o = np.argsort(v); c = np.cumsum(w[o]); c /= c[-1]; return np.interp(np.array(q) / 100, c, v[o])
for k in range(len(ids)):
    gz = (Z[k, 0] + Z[k, 1]).ravel(); z = Z[k, 2].ravel(); hz = np.histogram(z, zb, weights=gz)[0] / gz.sum()
    g = (P[k, 0] + P[k, 1]).ravel(); r = (P[k, 1].ravel() / np.maximum(g, 1e-3)); hp = np.histogram(r, pb, weights=g)[0] / g.sum()
    la = P[k, 2].ravel(); X.append(list(hz) + list(hp) + list(wq(la, g, qs)))
np.savez(_repo_io(f'hist-{split}.npz'), ids=np.array(ids), X=np.array(X), keys=np.array(keys)); print('done', split, np.array(X).shape)
