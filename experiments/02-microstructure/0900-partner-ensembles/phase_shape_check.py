"""Label-free: per-image, do dark (second-phase) grains differ from matrix grains in size, shape, neighbours or clustering? Clean train images."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]
Mc = np.load(_repo_path(E, 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); ids = [str(i) for i in Mc['ids']]
nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255
out = []
for n, i in enumerate(ids):
    if nz[n] >= 9.5: continue
    img = ndi.gaussian_filter(np.array(Image.open(_repo_path(P, f'data/train/{i}.png'))).astype(np.float64), 1.0); m = Mc['masks'][n].astype(int)
    er = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = measure.regionprops(m); L = np.array([p.label for p in props]); A = np.array([p.area for p in props], float)
    I = np.asarray(ndi.mean(img, er, L)); S = np.nanquantile(I, 0.75); z = 1 - I / S
    AR = np.array([p.axis_major_length / max(p.axis_minor_length, 1) for p in props]); SOL = np.array([p.solidity for p in props])
    inner = np.array([p.bbox[0] > 0 and p.bbox[1] > 0 and p.bbox[2] < 256 and p.bbox[3] < 256 for p in props])
    pore = (z > 0.6) & (SOL > 0.85); dark = (z > 0.15) & ~pore; mat = (z < 0.08)
    ok = inner & (A >= 12)
    if (dark & ok).sum() < 5 or (mat & ok).sum() < 10: continue
    # neighbour phase share: fraction of a dark grain's neighbours that are dark, vs dark share overall
    lut = np.zeros(m.max() + 1, int); lut[L[dark]] = 1; lut[L[mat]] = 2
    a, b = m[:, :-1].ravel(), m[:, 1:].ravel(); c, d = m[:-1, :].ravel(), m[1:, :].ravel()
    pa = np.r_[a, c]; pb = np.r_[b, d]; s = (pa != pb) & (pa > 0) & (pb > 0); pa, pb = pa[s], pb[s]
    pairs = np.unique(np.c_[np.minimum(pa, pb), np.maximum(pa, pb)], axis=0); ca, cb = lut[pairs[:, 0]], lut[pairs[:, 1]]
    dd = ((ca == 1) & (cb == 1)).sum(); dm = (((ca == 1) & (cb == 2)) | ((ca == 2) & (cb == 1))).sum(); mm = ((ca == 2) & (cb == 2)).sum()
    fd = dark.sum() / (dark.sum() + mat.sum()); exp_dd = fd ** 2 / (fd ** 2 + 2 * fd * (1 - fd) + (1 - fd) ** 2)
    out.append([np.median(np.log(A[dark & ok])) - np.median(np.log(A[mat & ok])), np.median(AR[dark & ok]) - np.median(AR[mat & ok]),
                np.median(SOL[dark & ok]) - np.median(SOL[mat & ok]), dd / max(dd + dm + mm, 1), exp_dd, fd])
o = np.array(out); print('images', len(o))
for j, nm in enumerate(['dlog area (dark - matrix)', 'dAR', 'dsolidity', 'dark-dark pair share', 'random expectation', 'dark grain share']):
    print(f'{nm:28s} median {np.median(o[:, j]):+.3f}  IQR {np.percentile(o[:, j], 25):+.3f} .. {np.percentile(o[:, j], 75):+.3f}')
print('dark-dark observed / random expectation (median ratio)', np.median(o[:, 3] / o[:, 4]).round(3))
