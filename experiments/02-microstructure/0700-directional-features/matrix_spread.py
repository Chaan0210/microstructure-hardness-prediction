"""Probe: between-grain spread of MATRIX grain mean intensities (clean rows; zero-shot masks), relative to matrix-pore span, vs hardness."""
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
out = np.full((500, 5), np.nan)
for n in range(500):
    if nz[n] >= 9: continue
    im = np.array(Image.open(_repo_io(f'../../data/train/{ids[n]}.png'))).astype(float); m = masks[mids.index(ids[n])].astype(np.int64)
    inner = (ndi.maximum_filter(m, 5) == ndi.minimum_filter(m, 5)) & (m > 0)
    labs = np.unique(m[inner])
    if len(labs) < 15: continue
    I = ndi.mean(im, m * inner, labs); Aa = ndi.sum(np.ones_like(im), m * inner, labs); keep = Aa >= 15; I, Aa = I[keep], Aa[keep]
    mat_lvl = np.median(I[I > np.percentile(I, 40)]); pore = np.percentile(ndi.gaussian_filter(im, 2), 0.2); span = mat_lvl - pore
    z = (mat_lvl - I) / span; mat = z < 0.18                                  # matrix grains
    Im = I[mat]
    if len(Im) < 10: continue
    out[n] = [Im.std(), np.median(np.abs(Im - np.median(Im))) * 1.4826, Im.std() / span, np.percentile(Im, 90) - np.percentile(Im, 10), mat_lvl]
s = ~np.isnan(out[:, 0])
ctrl = ['u_ph', 'z_f25', 'z_f50', 'het4_u', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac', 'obj_I_cv', 'grain_I_std_mean']
C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
for j, k in enumerate(['mat_std', 'mat_mad', 'mat_std_rel', 'mat_p90_p10', 'mat_level']):
    v = out[s, j]; rv = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]; print('%-12s n=%d mean %.2f sd %.2f partial corr %+.3f' % (k, s.sum(), v.mean(), v.std(), np.corrcoef(rv, ry)[0, 1]))
