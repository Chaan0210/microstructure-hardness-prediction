"""Probe: grain-boundary intensity profile vs distance to boundary (clean rows, zero-shot masks): depth and width, vs hardness."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
from PIL import Image
from scipy import ndimage as ndi
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mids = [str(i) for i in M['ids']]; masks = M['masks']
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X']); ids = [str(i) for i in F['ids']]
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
nz = X[:, keys.index('noise')] * 255
out = np.full((500, 7), np.nan)
for n in range(500):
    if nz[n] >= 9: continue
    im = np.array(Image.open(_repo_io(f'../../data/train/{ids[n]}.png'))).astype(float); m = masks[mids.index(ids[n])].astype(np.int64)
    if m.max() < 15: continue
    edge = (ndi.maximum_filter(m, 3) != ndi.minimum_filter(m, 3))
    d = ndi.distance_transform_edt(~edge)
    # restrict to matrix-like grains (exclude dark phase & pores) using grain means
    labs = np.arange(1, m.max() + 1); I = ndi.mean(im, m, labs); I = np.nan_to_num(I, nan=0)
    hi = np.percentile(I[I > 0], 60); mat_g = np.zeros(m.max() + 1, bool); mat_g[1:] = I > hi - 8
    sel = mat_g[m] | edge
    prof = [im[(np.round(d) == k) & sel].mean() for k in range(0, 7)]
    ref = prof[-1]
    depth = ref - prof[0]; half = next((k for k in range(1, 7) if ref - prof[k] < depth / 2), 6)
    out[n] = [depth, depth / max(ref - np.percentile(ndi.gaussian_filter(im, 2), 0.2), 1), ref - prof[1], ref - prof[2], half, prof[0], edge.mean()]
s = ~np.isnan(out[:, 0])
ctrl = ['u_ph', 'z_f25', 'z_f50', 'het4_u', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac', 'grain_eqd_mean', 'aw_inv_sqrt_d']
C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
for j, k in enumerate(['gb_depth', 'gb_depth_rel', 'gb_d1', 'gb_d2', 'gb_halfwidth', 'gb_min_level', 'gb_frac']):
    v = out[s, j]; rv = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]; print('%-13s n=%d mean %.2f sd %.2f | raw corr y %+.3f partial %+.3f' % (k, s.sum(), v.mean(), v.std(), np.corrcoef(v, y[s])[0, 1], np.corrcoef(rv, ry)[0, 1]))
np.save(_repo_io('gbprof-train.npy'), out)
