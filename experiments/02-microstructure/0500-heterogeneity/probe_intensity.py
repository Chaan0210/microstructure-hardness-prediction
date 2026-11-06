"""Probe: does per-grain matrix intensity (possible orientation proxy) explain the 035 residual on clean images?"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage import measure, filters
from PIL import Image
sys.path.insert(0, '.')
import resid035 as R
ids, y, B, G, P = R.load(); b = B.mean(0); r = y - b
nz = R.noise(); n = np.array([nz[i] for i in ids])
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mid = {str(i): k for k, i in enumerate(M['ids'])}
clean = n <= np.quantile(n, 0.4)
rows = []
for i in ids[clean]:
    im = np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(np.float32)
    mask = M['masks'][mid[i]]
    sm = filters.gaussian(im, 1.0, preserve_range=True)
    inner = mask * (ndi.minimum_filter(mask, 3) == ndi.maximum_filter(mask, 3))
    pr = measure.regionprops(inner, intensity_image=sm); pa = {p.label: p.area for p in measure.regionprops(mask)}
    A = np.array([pa[p.label] for p in pr], float); I = np.array([p.intensity_mean for p in pr]); AR = np.array([p.axis_major_length / max(p.axis_minor_length, 1) for p in pr])
    hi = np.percentile(I, 75); lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    mt = (z <= 0.3) & (A > 20)
    zm = I[mt]; am = A[mt]
    mu = np.average(zm, weights=am); sd = np.sqrt(np.average((zm - mu) ** 2, weights=am))
    q = (zm - mu) / max(sd, 1e-3)
    d = dict(id=i, n=len(A), mt_mean=mu, mt_sd=sd, mt_cv=sd / mu,
             mt_bright_frac=am[q > 1].sum() / am.sum(), mt_dark_frac=am[q < -1].sum() / am.sum(),
             mt_skew=np.average(q ** 3, weights=am), mt_kurt=np.average(q ** 4, weights=am),
             mt_rho_size=spearmanr(zm, am)[0], mt_rho_ar=spearmanr(zm, AR[mt])[0],
             mt_range=(zm.max() - zm.min()) / mu, mt_nlevels=len(np.unique(np.round(zm / 4))))
    rows.append(d)
rr = r[clean]; yy = y[clean]; cols = [k for k in rows[0] if k != 'id']
Z = {c: np.array([d[c] for d in rows], float) for c in cols}
co = Z['n'] < np.median(Z['n'])
for c in cols:
    print(f'{c:16s} resid rho all {spearmanr(Z[c], rr)[0]:+.3f}  coarse {spearmanr(Z[c][co], rr[co])[0]:+.3f}   y rho {spearmanr(Z[c], yy)[0]:+.3f}')
print('n clean', len(rr), 'coarse', co.sum())
