"""Graph correlogram of log grain area: normalised covariance between interior grains at graph distance 1..K (K=6), zero-shot Cellpose-SAM masks.
usage: graph3_feats.py SPLIT [MASKFILE] -> graph3-SPLIT.npz (keys gc1..gc6)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy import ndimage as ndi
from multiprocessing import Pool
split = sys.argv[1]; mf = sys.argv[2] if len(sys.argv) > 2 else f'../grain-cellpose-220/masks-v3-{split}.npz'
M = np.load(_repo_io(mf), allow_pickle=True); ids = [str(i) for i in M['ids']]; masks = M['masks']
K = 6; KEYS = [f'gc{k}' for k in range(1, K + 1)]
def one(m):
    m = m.astype(np.int32); labs = np.unique(m); labs = labs[labs > 0]
    if len(labs) < 20: return [np.nan] * K
    area = ndi.sum(np.ones(m.shape), m, labs); lut = dict(zip(labs.tolist(), np.log(area).tolist()))
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
    iset = set(inner); v = np.array([lut[l] for l in inner]); mu = v.mean(); s2 = v.var()
    acc = [[] for _ in range(K)]
    for s in inner:   # BFS up to depth K over the full graph, collect interior targets with s < t
        seen = {s}; front = {s}
        for d in range(K):
            nxt = set()
            for x in front: nxt |= nb.get(x, set())
            nxt -= seen; seen |= nxt; front = nxt
            for t in front:
                if t in iset and t > s: acc[d].append((lut[s] - mu) * (lut[t] - mu))
    return [np.mean(a) / s2 if len(a) > 5 else np.nan for a in acc]
if __name__ == '__main__':
    with Pool(8) as pool: X = pool.map(one, list(masks), chunksize=8)
    out = sys.argv[3] if len(sys.argv) > 3 else f'graph3-{split}.npz'
    np.savez(_repo_io(out), ids=np.array(ids), X=np.array(X, float), keys=np.array(KEYS)); print('done', split, np.nanmean(np.array(X, float), 0).round(4))
