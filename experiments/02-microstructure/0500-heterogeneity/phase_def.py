"""Clean half, exact masks: which phase definition (z threshold / soft) carries the most hardness information beyond non-phase structure?"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
sys.path.insert(0, '.')
import resid035 as R
lab = R.labels(); nz = R.noise()
F = np.load(_repo_io('../push-400/feat-u1dmavg-train.npz'), allow_pickle=True); k = [str(x) for x in F['keys']]; ids = [str(i) for i in F['ids']]
y = np.array([lab[i] for i in ids]); n = np.array([nz[i] for i in ids]) * 255
ctrl = ['m_la_mean', 'm_la_f32', 'm_la_tile8', 'm_el_glob', 'm_el_pix', 'stcoh_s3_w8_locmed', 'stcoh_s1.5_w32_glob', 'pore_frac', 'm_la_std']
Xc = F['X'][:, [k.index(c) for c in ctrl]].astype(float); Xc = np.where(np.isnan(Xc), np.nanmedian(Xc, 0), Xc); C = np.c_[np.ones(len(y)), Xc]
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mid = {str(i): j for j, i in enumerate(M['ids'])}
sel = np.where(n < np.median(n))[0]
TH = [0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.33, 0.36, 0.4]
V = {f'f{t}': [] for t in TH}; V.update({f's{t}': [] for t in (0.2, 0.25, 0.3)}); V['zlin'] = []; V['nfrac25'] = []; keep = []
for j in sel:
    i = ids[j]; a = np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(float); m = M['masks'][mid[i]].astype(np.int64)
    sm = ndi.gaussian_filter(a / 255, 1.0); inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}; ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = [l for l in props if l in ipr and ipr[l].area >= 3]
    if len(labs) < 8: continue
    A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1) for l in labs]); SOL = np.array([props[l].solidity for l in labs])
    o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3 * np.median(A)); g = ~pore; Ag, zg = A[g], z[g]
    for t in TH: V[f'f{t}'].append(Ag[zg > t].sum() / Ag.sum())
    for t in (0.2, 0.25, 0.3): V[f's{t}'].append((Ag / (1 + np.exp(-(zg - t) / 0.03))).sum() / Ag.sum())
    V['zlin'].append((Ag * np.clip((zg - 0.15) / 0.2, 0, 1)).sum() / Ag.sum()); V['nfrac25'].append((zg > 0.25).mean())
    keep.append(j)
keep = np.array(keep)
def pcorr(v):
    A_ = C[keep]; ry = y[keep] - A_ @ np.linalg.lstsq(A_, y[keep], rcond=None)[0]; rv = v - A_ @ np.linalg.lstsq(A_, v, rcond=None)[0]; return np.corrcoef(rv, ry)[0, 1]
print('n', len(keep))
U = {'u_ph_hard': F['X'][keep, k.index('u_ph_hard')], 'm_ph_hard': F['X'][keep, k.index('m_ph_hard')], 'u_ph': F['X'][keep, k.index('u_ph')]}
for nm, v in list(V.items()) + list(U.items()):
    v = np.array(v, float); print(f'{nm:10s} raw corr {np.corrcoef(v, y[keep])[0, 1]:+.3f}  pcorr {pcorr(v):+.3f}')
