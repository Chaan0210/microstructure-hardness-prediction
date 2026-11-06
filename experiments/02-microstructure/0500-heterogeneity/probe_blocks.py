"""Clean rows, exact masks: which block-statistic form of local grain size relates to the 035 residual? (exploratory)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage import measure
sys.path.insert(0, '.')
import resid035 as R
ids, y, B, G, P = R.load(); r = y - B.mean(0); nz = R.noise(); n = np.array([nz[i] for i in ids]) * 255
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mid = {str(i): k for k, i in enumerate(M['ids'])}
sel = np.where(n < 9.5)[0]
F = {}
for k in sel:
    i = ids[k]; a = np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(float); m = M['masks'][mid[i]].astype(np.int64)
    sm = ndi.gaussian_filter(a / 255, 1.0); inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}; ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = np.array([l for l in props if l in ipr and ipr[l].area >= 3])
    if len(labs) < 8: continue
    A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1) for l in labs]); SOL = np.array([props[l].solidity for l in labs])
    o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3 * np.median(A)); g = ~pore; ph = g & (z > 0.3)
    lut = np.zeros(m.max() + 1); lut[labs[g]] = A[g]; amap = lut[m]; valid = amap > 0
    plut = np.zeros(m.max() + 1); plut[labs[ph]] = 1; pmap = plut[m]
    d = {}
    for nb in (2, 4, 8):
        s = 256 // nb
        def bs(v): return v.reshape(nb, s, nb, s).sum((1, 3))
        den = bs(valid.astype(float)); ok = den > 0.2 * s * s
        bh = -2 * np.log(bs(np.where(valid, np.maximum(amap, 1) ** -0.5, 0))[ok] / den[ok])   # HP-weighted
        bl = bs(np.where(valid, np.log(np.maximum(amap, 1)), 0))[ok] / den[ok]                 # mean log area (pixel-weighted)
        bq = -4 * np.log(bs(np.where(valid, np.maximum(amap, 1) ** -0.25, 0))[ok] / den[ok])  # d^-1/2 weighted
        pf = bs(pmap * valid)[ok] / den[ok]
        for nm, b in (('hp', bh), ('lg', bl), ('q', bq)):
            d[f'{nm}{nb}_std'] = b.std(); d[f'{nm}{nb}_max'] = b.max() - b.mean(); d[f'{nm}{nb}_min'] = b.mean() - b.min(); d[f'{nm}{nb}_rng'] = b.max() - b.min()
        d[f'pf{nb}_std'] = pf.std(); d[f'corr_pf_size{nb}'] = np.corrcoef(pf, bh)[0, 1] if pf.std() > 0 else 0
    F[k] = d
ks = list(F); keys = list(F[ks[0]])
rr = r[ks]; yy = y[ks]; la = np.array([F[k]['lg4_std'] for k in ks])
print('n', len(ks))
res = []
for key in keys:
    v = np.array([F[k][key] for k in ks]); v = np.nan_to_num(v)
    res.append((key, spearmanr(v, yy)[0], spearmanr(v, rr)[0]))
for key, a, b in sorted(res, key=lambda t: -abs(t[2]))[:25]: print(f'{key:16s} y {a:+.3f} resid {b:+.3f}')
