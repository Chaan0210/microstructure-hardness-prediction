"""Grain-adjacency-graph statistics of log grain area from zero-shot Cellpose-SAM masks (grain-cellpose-220/masks-v3-SPLIT.npz). Label-free.
mI1/mI2/mI3: Moran-type correlation of log area between 1st / 2nd / 3rd-order neighbour pairs (interior grains), cov*: unnormalised covariances.
usage: graph_feats.py SPLIT -> graph-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy import ndimage as ndi
from multiprocessing import Pool
split = sys.argv[1]
M = np.load(_repo_io(f'../grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); ids = [str(i) for i in M['ids']]; masks = M['masks']
KEYS = ['mI1', 'mI2', 'mI3', 'cov1', 'cov2', 'cov3', 'la_var_in', 'n_pairs1']
def one(m):
    m = m.astype(np.int32); labs = np.unique(m); labs = labs[labs > 0]
    if len(labs) < 20: return [np.nan] * len(KEYS)
    area = ndi.sum(np.ones(m.shape), m, labs); la = np.log(area); lut = np.zeros(m.max() + 1); lut[labs] = la
    border = set(np.unique(np.r_[m[0], m[-1], m[:, 0], m[:, -1]]).tolist())
    pairs = set()
    for dy, dx in [(0, 3), (3, 0), (3, 3), (3, -3)]:
        a = m[max(0, -dy):m.shape[0] - max(0, dy), max(0, -dx):m.shape[1] - max(0, dx)]
        b = m[max(0, dy):, max(0, dx):][:a.shape[0], :a.shape[1]]
        sel = (a > 0) & (b > 0) & (a != b); pr = np.sort(np.stack([a[sel], b[sel]], 1), 1); pairs |= set(map(tuple, np.unique(pr, axis=0).tolist()))
    nb = {}
    for a, b in pairs: nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
    inner = [l for l in labs.tolist() if l not in border]
    if len(inner) < 8: inner = labs.tolist()
    inner_set = set(inner); v = lut[inner]; mu = v.mean(); s2 = v.var()
    # graph distances up to 3 by BFS from each interior grain
    d1, d2, d3 = set(), set(), set()
    for s in inner:
        f1 = nb.get(s, set()); f2 = set().union(*[nb.get(x, set()) for x in f1]) - f1 - {s} if f1 else set()
        f3 = set().union(*[nb.get(x, set()) for x in f2]) - f2 - f1 - {s} if f2 else set()
        for S, D in ((f1, d1), (f2, d2), (f3, d3)):
            for t in S:
                if t in inner_set: D.add((min(s, t), max(s, t)))
    out = []
    covs = []
    for D in (d1, d2, d3):
        if len(D) < 5: covs.append(np.nan); continue
        P = np.array(list(D)); covs.append(float(np.mean((lut[P[:, 0]] - mu) * (lut[P[:, 1]] - mu))))
    return [c / s2 for c in covs] + covs + [s2, len(d1)]
if __name__ == '__main__':
    with Pool(8) as pool: X = pool.map(one, list(masks), chunksize=8)
    np.savez(_repo_io(f'graph-{split}.npz'), ids=np.array(ids), X=np.array(X, float), keys=np.array(KEYS)); print('done', split, np.nanmean(np.array(X, float), 0))
