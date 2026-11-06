"""Label-free: do second-phase grains carry an internal texture (beyond white noise) that could identify them when brightness contrast is lost?
Clean train images (noise < 9.5), masks-v3. Per grain (2-px eroded interior, >= 40 px): residual after removing the grain mean; compare
residual SD and lag-1/lag-2 autocorrelation and spectral share for dark (z > 0.15) vs matrix (z < 0.05) grains, relative to the image's
white-noise SD (MAD of the 2nd difference)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]
Mc = np.load(_repo_path(E, 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); ids = [str(i) for i in Mc['ids']]
nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255
yy, xx = np.mgrid[0:256, 0:256] / 255.0 - 0.5
rows = {'dark': [], 'matrix': []}
for n, i in enumerate(ids):
    if nz[n] >= 9.5: continue
    img = np.array(Image.open(_repo_path(P, f'data/train/{i}.png'))).astype(np.float64); m = Mc['masks'][n].astype(int)
    d2 = img[:, 2:] - 2 * img[:, 1:-1] + img[:, :-2]; sw = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)
    er = m * (ndi.minimum_filter(m, 5) == ndi.maximum_filter(m, 5))
    L = np.unique(er); L = L[L > 0]; A = np.asarray(ndi.sum(np.ones_like(img), er, L)); I = np.asarray(ndi.mean(img, er, L))
    S = np.quantile(I, 0.75); z = 1 - I / S
    for l, a, zz in zip(L, A, z):
        if a < 40: continue
        sel = er == l; v = img[sel] - img[sel].mean(); sd = v.std()
        # lag-1 horizontal autocorrelation inside the grain
        sh = sel[:, 1:] & sel[:, :-1]; r1 = np.corrcoef(img[:, 1:][sh], img[:, :-1][sh])[0, 1] if sh.sum() > 20 else np.nan
        cls = 'dark' if zz > 0.15 else ('matrix' if zz < 0.05 else None)
        if cls: rows[cls].append((sd / sw, r1, a))
for k, v in rows.items():
    v = np.array(v); print(f'{k:6s} n={len(v):6d}  SD/white {np.nanmedian(v[:, 0]):.3f} (IQR {np.nanpercentile(v[:, 0], 25):.2f}-{np.nanpercentile(v[:, 0], 75):.2f})  lag1 r {np.nanmedian(v[:, 1]):.3f}')
