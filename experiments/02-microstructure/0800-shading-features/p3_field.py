"""P3: is the extra per-grain intensity scatter of real noisy images spatially structured (correctable illumination / etch field) or iid?
Matrix grains = U-Net(z1) phase prob < 0.15 and pore prob < 0.1 over the grain. Offsets r_i = I_i - median_matrix.
Stats per image: neighbour-pair correlation of r, R^2 of r on aligned 4x4 zone dummies vs 32px-shifted grid, smooth-field R^2 (quadratic surface)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
from scipy import ndimage as ndi
from PIL import Image
P = str(_REPO_ROOT) + '/'; E = P + 'experiments/'
U = np.load(_repo_io(E + 'push-400/unet4maps-z1-train.npz')); upos = {str(i): k for k, i in enumerate(U['ids'])}
Mc = np.load(_repo_io(E + 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); cpos = {str(i): k for k, i in enumerate(Mc['ids'])}
Mn = np.load(_repo_io(E + 'push-300/ftseg-ft-H300-train-noisy-n1-masks.npz'), allow_pickle=True); npos = {str(i): k for k, i in enumerate(Mn['ids'])}
F = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float); ids = [str(i) for i in F['ids']]
nz = X[:, keys.index('noise')] * 255
def zone_r2(r, cx, cy, off):
    zx = np.clip(((cx + off) // 64).astype(int), 0, 4); zy = np.clip(((cy + off) // 64).astype(int), 0, 4); z = zx * 5 + zy
    tot = np.sum((r - r.mean()) ** 2); within = sum(np.sum((r[z == u] - r[z == u].mean()) ** 2) for u in np.unique(z))
    k = len(np.unique(z)); n = len(r)
    return 1 - within / tot - (k - 1) / (n - 1)   # chance-adjusted
out = {'clean': [], 'noisy': []}
for k, i in enumerate(ids):
    if not np.isfinite(nz[k]): continue
    if nz[k] < 7.0: grp, mask = 'clean', Mc['masks'][cpos[i]].astype(int)
    elif nz[k] >= 11.9 and i in npos: grp, mask = 'noisy', Mn['masks'][npos[i]].astype(int)
    else: continue
    im = np.asarray(Image.open(_repo_io(P + f'data/train/{i}.png')), np.float32)
    mp = U['maps'][upos[i]].astype(np.float32); ph = np.kron(mp[1], np.ones((2, 2))); po = np.kron(mp[2], np.ones((2, 2)))
    inner = np.where(ndi.minimum_filter(mask, 5) == ndi.maximum_filter(mask, 5), mask, 0)
    labs = np.unique(inner); labs = labs[labs > 0]
    if len(labs) < 25: continue
    I = np.asarray(ndi.mean(im, inner, labs)); pph = np.asarray(ndi.mean(ph, inner, labs)); ppo = np.asarray(ndi.mean(po, inner, labs))
    cnt = np.asarray(ndi.sum(np.ones_like(im), inner, labs)); com = np.array(ndi.center_of_mass(np.ones_like(im), inner, labs))
    m = (pph < 0.15) & (ppo < 0.1) & (cnt >= 30)
    if m.sum() < 20: continue
    r = I[m] - np.median(I[m]); cy, cx = com[m, 0], com[m, 1]
    # neighbour pairs by centroid distance
    d = np.hypot(cx[:, None] - cx[None], cy[:, None] - cy[None]); s = np.sqrt(65536 / len(labs))
    a, b = np.where((d < 1.3 * s) & (d > 0)); nc = np.corrcoef(r[a], r[b])[0, 1] if len(a) > 10 else np.nan
    far_a, far_b = np.where(d > 4 * s); fc = np.corrcoef(r[far_a], r[far_b])[0, 1] if len(far_a) > 10 else np.nan
    A = np.c_[np.ones(len(r)), cx, cy, cx ** 2, cy ** 2, cx * cy] / np.r_[1, 256, 256, 65536, 65536, 65536]
    res = r - A @ np.linalg.lstsq(A, r, rcond=None)[0]; quad = 1 - res.var() / r.var() - 5 / (len(r) - 1)
    out[grp].append((r.std(), nc, fc, zone_r2(r, cx, cy, 0), zone_r2(r, cx, cy, 32), quad, m.sum()))
for g, v in out.items():
    v = np.array(v, float)
    print(f'{g:6s} n_img {len(v):3d} | matrix-grain offset SD {np.nanmedian(v[:, 0]):.2f} gray | neighbour corr {np.nanmean(v[:, 1]):+.3f} (far {np.nanmean(v[:, 2]):+.3f}) | '
          f'zone R2 aligned {np.nanmean(v[:, 3]):+.3f} shifted32 {np.nanmean(v[:, 4]):+.3f} | quadratic-surface R2 {np.nanmean(v[:, 5]):+.3f} | matrix grains/img {np.median(v[:, 6]):.0f}')
