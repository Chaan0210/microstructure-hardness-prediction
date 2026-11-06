"""Clean rows (noise<9.5), exact zero-shot masks: broad battery of grain statistics vs the 038 (H2Z) dev residual. Exploratory."""
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
ids, y, B, G, P = R.load(tags=['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']); r = y - B.mean(0)
nz = R.noise(); n = np.array([nz[i] for i in ids]) * 255
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mid = {str(i): j for j, i in enumerate(M['ids'])}
sel = np.where(n < 9.5)[0]; rows = {}
import os
if os.environ.get('COARSE'): pass
for j in sel:
    i = ids[j]; a = np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(float); m = M['masks'][mid[i]].astype(np.int64)
    sm = ndi.gaussian_filter(a / 255, 1.0); inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}; ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = np.array([l for l in props if l in ipr and ipr[l].area >= 3])
    if len(labs) < 8: continue
    A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1) for l in labs]); SOL = np.array([props[l].solidity for l in labs])
    PER = np.array([props[l].perimeter for l in labs]); TH = np.array([props[l].orientation for l in labs]); EC = np.array([props[l].eccentricity for l in labs])
    bb = np.array([props[l].bbox for l in labs]); B_ = (bb[:, 0] == 0) | (bb[:, 1] == 0) | (bb[:, 2] == 256) | (bb[:, 3] == 256)
    o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3 * np.median(A)); g = ~pore; ph = g & (z > 0.25); mx = g & ~ph
    d = {}
    la = np.log(A[g]); w = A[g] / A[g].sum()
    d['la_nw_mean'] = la.mean(); d['la_aw_mean'] = (w * la).sum(); d['la_nw_sd'] = la.std(); d['la_aw_sd'] = np.sqrt((w * (la - d['la_aw_mean']) ** 2).sum())
    d['la_q10'] = np.quantile(la, 0.1); d['la_q90'] = np.quantile(la, 0.9); d['la_max'] = la.max(); d['top5_share'] = np.sort(A[g])[-5:].sum() / A[g].sum()
    d['simpson_neff'] = 1 / (w ** 2).sum(); d['log_simpson'] = np.log(d['simpson_neff']); d['n_border_frac'] = B_[g].mean(); d['border_area_frac'] = A[g & B_].sum() / A[g].sum()
    d['ph_la_mean'] = np.log(A[ph]).mean() if ph.sum() else 0; d['mx_la_mean'] = np.log(A[mx]).mean(); d['ph_mx_la'] = d['ph_la_mean'] - d['mx_la_mean']
    d['ph_ar'] = np.average(np.log(AR[ph]), weights=A[ph]) if ph.sum() else 0; d['mx_ar'] = np.average(np.log(AR[mx]), weights=A[mx])
    d['ph_nfrac'] = ph.sum() / g.sum(); d['ph_afrac'] = A[ph].sum() / A[g].sum(); d['ph_large_frac'] = A[ph & (A > np.median(A))].sum() / A[g].sum()
    d['perim_density'] = PER[g].sum() / 65536; d['sol_aw'] = (w * SOL[g]).sum(); d['ecc_aw'] = (w * EC[g]).sum()
    d['ar_q90'] = np.quantile(AR[g], 0.9); d['logar_aw'] = (w * np.log(AR[g])).sum(); d['logar_sd'] = np.log(AR[g]).std()
    wt = A[g] * np.log(AR[g]); c2 = (wt * np.cos(2 * TH[g])).sum() / wt.sum(); s2 = (wt * np.sin(2 * TH[g])).sum() / wt.sum(); d['S_align'] = np.hypot(c2, s2); d['S_x_ar'] = d['S_align'] * d['logar_aw']
    d['ang'] = 0.5 * np.arctan2(s2, c2); d['cos2ang'] = np.cos(2 * d['ang'])
    d['pore_n'] = pore.sum(); d['pore_af'] = A[pore].sum() / 65536; d['pore_in_ph'] = 0.0
    d['I_mx_sd'] = np.sqrt(np.average((I[mx] - np.average(I[mx], weights=A[mx])) ** 2, weights=A[mx])) / max(hi - lo, 1e-3)
    d['z_ph_mean'] = np.average(z[ph], weights=A[ph]) if ph.sum() else 0
    rows[j] = d
ks = np.array(list(rows)); keys = list(rows[ks[0]])
rr = r[ks]
res = []
for kk in keys:
    v = np.array([rows[j][kk] for j in ks]); res.append((kk, spearmanr(v, rr)[0], spearmanr(v, y[ks])[0]))
print('n', len(ks))
for kk, a, b in sorted(res, key=lambda t: -abs(t[1])): print(f'{kk:16s} resid {a:+.3f}  y {b:+.3f}')
