"""Exact structure statistics from zero-shot masks-v3 (clean rows are reliable). Label-free, fixed definitions.
fd_area: second-phase area fraction (features.py z>0.3 rule) | pore_frac | logasp_aw: area-weighted mean log aspect ratio (non-pore)
S_align: area*(AR-1)-weighted doubled-angle order parameter | logNeff: log(interior + 0.5*border non-pore grains)
het4_px / het8_px: std over 4x4 / 8x8 blocks of b=-2 log(mean over non-pore grain pixels of a(p)^-1/2)  (pixel-weighted)
het4_c: same but grains assigned by centroid (our old het_blocks.py style) | bd_frac: border-touching share of non-pore grains
usage: struct_feats.py SPLIT"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
H = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
split = sys.argv[1]
M = np.load(_repo_path(H.parent, f'grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True)
ids = [str(i) for i in M['ids']]
KEYS = ['fd_area', 'pore_frac_m', 'logasp_aw', 'S_align', 'logNeff', 'het4_px', 'het8_px', 'het4_c', 'bd_frac', 'la_aw', 'la_nw', 'n_grains']
def blocks(v, nb):
    s = 256 // nb; return v.reshape(nb, s, nb, s).sum((1, 3))
X = []
for n, i in enumerate(ids):
    a = np.array(Image.open(_repo_path(P, f'data/{split}/{i}.png'))).astype(float); m = M['masks'][n].astype(np.int64)
    sm = ndi.gaussian_filter(a / 255, 1.0); inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}; ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = np.array([l for l in props if l in ipr and ipr[l].area >= 3])
    if len(labs) < 8:
        X.append([np.nan] * len(KEYS)); continue
    A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1) for l in labs]); SOL = np.array([props[l].solidity for l in labs])
    TH = np.array([props[l].orientation for l in labs]); CY = np.array([props[l].centroid[0] for l in labs]); CX = np.array([props[l].centroid[1] for l in labs])
    bb = np.array([props[l].bbox for l in labs]); B = (bb[:, 0] == 0) | (bb[:, 1] == 0) | (bb[:, 2] == 256) | (bb[:, 3] == 256)
    o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3 * np.median(A)); g = ~pore
    d = {}
    d['fd_area'] = A[g & (z > 0.3)].sum() / A[g].sum(); d['pore_frac_m'] = A[pore].sum() / 65536
    d['logasp_aw'] = np.sum(A[g] * np.log(AR[g])) / A[g].sum()
    w = A[g] * (AR[g] - 1); d['S_align'] = np.hypot(np.sum(w * np.cos(2 * TH[g])), np.sum(w * np.sin(2 * TH[g]))) / max(w.sum(), 1e-6)
    neff = (g & ~B).sum() + 0.5 * (g & B).sum(); d['logNeff'] = np.log(max(neff, 1)); d['n_grains'] = float(g.sum())
    d['bd_frac'] = (g & B).sum() / max(g.sum(), 1)
    d['la_aw'] = np.sum(A[g] * np.log(A[g])) / A[g].sum(); d['la_nw'] = np.mean(np.log(A[g]))
    # pixel maps: area of containing grain for non-pore grain pixels
    lut = np.zeros(m.max() + 1); lut[labs[g]] = A[g]; amap = lut[m]; valid = amap > 0
    inv = np.where(valid, np.maximum(amap, 1) ** -0.5, 0.0)
    for nb, key in ((4, 'het4_px'), (8, 'het8_px')):
        num = blocks(inv, nb); den = blocks(valid.astype(float), nb); ok = den > 0.2 * (256 // nb) ** 2
        b = -2 * np.log(num[ok] / den[ok]); d[key] = float(b.std()) if ok.sum() > 3 else np.nan
    ty = np.minimum((CY[g] // 64).astype(int), 3); tx = np.minimum((CX[g] // 64).astype(int), 3); bc = []
    for p in range(4):
        for q in range(4):
            s = (ty == p) & (tx == q)
            if s.sum() >= 3: bc.append(-2 * np.log(np.mean(A[g][s] ** -0.5)))
    d['het4_c'] = float(np.std(bc)) if len(bc) > 3 else np.nan
    X.append([float(d[k]) for k in KEYS])
    if (n + 1) % 100 == 0: print(n + 1, flush=True)
np.savez(_repo_path(H, f'struct-{split}.npz'), ids=np.array(ids), X=np.array(X), keys=np.array(KEYS)); print('done')
