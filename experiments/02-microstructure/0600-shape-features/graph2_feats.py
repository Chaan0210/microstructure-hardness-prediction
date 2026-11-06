"""Affine-invariant tessellation-regularity statistics from zero-shot Cellpose-SAM masks (label-free).
deg_*: neighbour-count distribution of interior grains; lewis: corr(log area, degree); aw: Aboav-Weaire corr(degree, mean neighbour degree);
la_cv: CV of area; la_kurt/la_skew of log area; mI2 sanity.  usage: graph2_feats.py SPLIT -> graph2-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
from multiprocessing import Pool
split = sys.argv[1]
M = np.load(_repo_io(f'../grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); ids = [str(i) for i in M['ids']]; masks = M['masks']
KEYS = ['deg_mean', 'deg_std', 'deg_f6', 'lewis', 'aw', 'a_cv', 'la_skew', 'la_kurt', 'mI1n', 'mI2n']
def one(m):
    m = m.astype(np.int32); labs = np.unique(m); labs = labs[labs > 0]
    if len(labs) < 20: return [np.nan] * len(KEYS)
    area = ndi.sum(np.ones(m.shape), m, labs); amap = dict(zip(labs.tolist(), area.tolist()))
    border = set(np.unique(np.r_[m[0], m[-1], m[:, 0], m[:, -1]]).tolist())
    pairs = set()
    for dy, dx in [(0, 3), (3, 0), (3, 3), (3, -3)]:
        a = m[max(0, -dy):m.shape[0] - max(0, dy), max(0, -dx):m.shape[1] - max(0, dx)]
        b = m[max(0, dy):, max(0, dx):][:a.shape[0], :a.shape[1]]
        sel = (a > 0) & (b > 0) & (a != b); pr = np.sort(np.stack([a[sel], b[sel]], 1), 1); pairs |= set(map(tuple, np.unique(pr, axis=0).tolist()))
    nb = {}
    for a, b in pairs: nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
    # interior = not touching border and none of its neighbours is missing (all neighbours known): use not-border and degree>=3
    inner = [l for l in labs.tolist() if l not in border and len(nb.get(l, ())) >= 3]
    if len(inner) < 10: return [np.nan] * len(KEYS)
    deg = np.array([len(nb[l]) for l in inner], float); la = np.log([amap[l] for l in inner]); A = np.exp(la)
    mnd = np.array([np.mean([len(nb[x]) for x in nb[l]]) for l in inner])
    mu = la.mean(); s2 = la.var(); lut = {l: v for l, v in zip(inner, la)}
    p1 = [(a, b) for a, b in pairs if a in lut and b in lut]
    mI1 = np.mean([(lut[a] - mu) * (lut[b] - mu) for a, b in p1]) / s2 if len(p1) > 5 else np.nan
    p2 = set()
    for c in inner:
        ns = [x for x in nb[c] if x in lut]
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                if ns[j] not in nb[ns[i]]: p2.add((min(ns[i], ns[j]), max(ns[i], ns[j])))
    mI2 = np.mean([(lut[a] - mu) * (lut[b] - mu) for a, b in p2]) / s2 if len(p2) > 5 else np.nan
    return [deg.mean(), deg.std(), np.mean(deg == 6), np.corrcoef(la, deg)[0, 1], np.corrcoef(deg, mnd)[0, 1], A.std() / A.mean(), skew(la), kurtosis(la), mI1, mI2]
if __name__ == '__main__':
    with Pool(8) as pool: X = pool.map(one, list(masks), chunksize=8)
    np.savez(_repo_io(f'graph2-{split}.npz'), ids=np.array(ids), X=np.array(X, float), keys=np.array(KEYS)); print('done', split, np.nanmean(np.array(X, float), 0).round(3))
