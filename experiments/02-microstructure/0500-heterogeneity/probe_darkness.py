"""Probe on clean images (zero-shot masks): is hardness tied to binary phase area fraction or to a continuous darkness measure?"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr, pearsonr
from skimage import measure
sys.path.insert(0, '.')
import resid035 as R
lab = R.labels(); nz = R.noise()
ids_all = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/train.csv')))]
n_all = np.array([nz[i] for i in ids_all]); clean_ids = [i for i in ids_all if nz[i] < np.quantile(n_all, 0.4)]
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mid = {str(i): k for k, i in enumerate(M['ids'])}
rid, y, B, G, P = R.load(); res = dict(zip(rid, y - B.mean(0)))
rows = {}
for i in clean_ids:
    a = np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(float); m = M['masks'][mid[i]].astype(np.int64)
    sm = ndi.gaussian_filter(a / 255, 1.0); inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}; ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = [l for l in props if l in ipr and ipr[l].area >= 3]
    A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1) for l in labs]); SOL = np.array([props[l].solidity for l in labs])
    o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3 * np.median(A)); g = ~pore; Ag = A[g]; zg = z[g]
    d = {}
    for t in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5): d[f'f{int(t*100)}'] = Ag[zg > t].sum() / Ag.sum()
    d['zmean'] = (Ag * zg).sum() / Ag.sum(); d['zpos'] = (Ag * np.clip(zg, 0, None)).sum() / Ag.sum()
    ph = zg > 0.3
    d['ph_depth'] = (Ag[ph] * zg[ph]).sum() / max(Ag[ph].sum(), 1); d['zw_ph'] = (Ag[ph] * zg[ph]).sum() / Ag.sum()
    d['nfrac30'] = ph.mean()
    # pixel-level fraction of image area covered by phase (incl. boundaries share)
    d['ph_px'] = Ag[ph].sum() / 65536
    d['pore_px'] = A[pore].sum() / 65536
    rows[i] = d
keys = list(next(iter(rows.values())).keys())
yy = np.array([lab[i] for i in clean_ids]); rr = np.array([res.get(i, np.nan) for i in clean_ids]); ok = ~np.isnan(rr)
print('n', len(clean_ids), 'with resid', ok.sum())
for k in keys:
    v = np.array([rows[i][k] for i in clean_ids])
    print(f'{k:10s} pearson y {pearsonr(v, yy)[0]:+.3f}  spearman y {spearmanr(v, yy)[0]:+.3f}  resid {spearmanr(v[ok], rr[ok])[0]:+.3f}')
np.savez(_repo_io('probe_darkness.npz'), ids=clean_ids, keys=keys, X=np.array([[rows[i][k] for k in keys] for i in clean_ids]), y=yy)
