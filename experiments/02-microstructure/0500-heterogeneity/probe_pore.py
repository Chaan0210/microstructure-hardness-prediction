"""Clean half, exact masks: are very dark compact objects (blurred small pores) counted as phase by the z>0.3 rule? Partial corr of alternative pore/phase splits."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
sys.path.insert(0, '.')
import resid035 as R
lab = R.labels(); nz = R.noise()
F = np.load(_repo_io('../push-400/feat-u1dmavg-train.npz'), allow_pickle=True); k = [str(x) for x in F['keys']]; ids = [str(i) for i in F['ids']]
y = np.array([lab[i] for i in ids]); n = np.array([nz[i] for i in ids]) * 255
ctrl = ['m_la_mean', 'm_la_f32', 'm_la_tile8', 'm_el_glob', 'm_el_pix', 'stcoh_s3_w8_locmed', 'stcoh_s1.5_w32_glob', 'm_la_std', 'm_ph_hard', 'u_ph_hard', 'u_ph']
Xc = F['X'][:, [k.index(c) for c in ctrl]].astype(float); Xc = np.where(np.isnan(Xc), np.nanmedian(Xc, 0), Xc); C = np.c_[np.ones(len(y)), Xc]
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mid = {str(i): j for j, i in enumerate(M['ids'])}
sel = np.where(n < np.median(n))[0]; V = {}; keep = []
for j in sel:
    i = ids[j]; a = np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(float); m = M['masks'][mid[i]].astype(np.int64)
    sm = ndi.gaussian_filter(a / 255, 1.0); inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}; ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = [l for l in props if l in ipr and ipr[l].area >= 3]
    if len(labs) < 8: continue
    A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs]); Imin = np.array([ipr[l].intensity_min for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1) for l in labs]); SOL = np.array([props[l].solidity for l in labs])
    o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    medA = np.median(A); pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3 * medA)
    compact = (SOL > 0.9) & (AR < 1.8)
    d = {}
    d['pore_rule'] = A[pore].sum() / 65536
    for t in (0.45, 0.55, 0.65):
        d[f'darkcompact{t}'] = A[(z > t) & compact & (A < 2 * medA)].sum() / 65536
        d[f'dark{t}'] = A[(z > t) & ~pore].sum() / A[~pore].sum()
    d['n_pore_rule'] = pore.sum(); d['n_darkcompact0.55'] = ((z > 0.55) & compact & (A < 2 * medA)).sum()
    V[j] = d; keep.append(j)
keep = np.array(keep)
def pc(v):
    A_ = C[keep]; ry = y[keep] - A_ @ np.linalg.lstsq(A_, y[keep], rcond=None)[0]; rv = v - A_ @ np.linalg.lstsq(A_, v, rcond=None)[0]; return np.corrcoef(rv, ry)[0, 1]
for kk in V[keep[0]]:
    v = np.array([V[j][kk] for j in keep], float); print(f'{kk:18s} raw corr {np.corrcoef(v, y[keep])[0, 1]:+.3f}  pcorr beyond phase+structure {pc(v):+.3f}  mean {v.mean():.4f}')
