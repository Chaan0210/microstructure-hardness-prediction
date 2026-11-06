"""Probe: grain-boundary darkness / width relative to matrix (clean rows), vs hardness."""
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
out = np.full((500, 4), np.nan)
for n in range(500):
    if nz[n] >= 11.9: continue
    im = np.array(Image.open(_repo_io(f'../../data/train/{ids[n]}.png'))).astype(float); m = masks[mids.index(ids[n])].astype(np.int64)
    edge = (ndi.maximum_filter(m, 3) != ndi.minimum_filter(m, 3))           # pixels next to a label change (boundary band ~2px)
    inner = (ndi.maximum_filter(m, 7) == ndi.minimum_filter(m, 7)) & (m > 0)
    labs = np.unique(m[inner])
    if len(labs) < 10: continue
    I = ndi.mean(im, m * inner, labs); Aa = ndi.sum(np.ones_like(im), m * inner, labs)
    mat = np.median(I[I > np.percentile(I, 50)]); pore = np.percentile(ndi.gaussian_filter(im, 2), 0.2)
    b = im[edge]; bmin = ndi.minimum_filter(ndi.gaussian_filter(im, 0.7), 3)[edge]
    out[n] = [(mat - np.median(b)) / (mat - pore), (mat - np.median(bmin)) / (mat - pore), edge.mean(), (mat - np.median(b))]
nm_ = ['bnd_dark_rel', 'bnd_min_rel', 'bnd_frac', 'bnd_dark_abs']
s = ~np.isnan(out[:, 0])
ctrl = ['u_ph', 'z_f25', 'z_f50', 'het4_u', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'sizefield_std_s32', 'pore_frac', 'grain_eqd_mean']
C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
for j, k in enumerate(nm_):
    v = out[s, j]; rv = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    print('%-14s mean %.3f sd %.3f | partial corr y (HQ) %+.3f' % (k, v.mean(), v.std(), np.corrcoef(rv, ry)[0, 1]))
