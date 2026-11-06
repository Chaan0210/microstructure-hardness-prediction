"""Probe: which heterogeneity definition (computed from realized grains, zero-shot masks, clean rows) best tracks hardness."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
from scipy import ndimage as ndi
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mids = [str(i) for i in M['ids']]; masks = M['masks']
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X']); ids = [str(i) for i in F['ids']]
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
nz = X[:, keys.index('noise')] * 255
defs = ['cv_area', 'cv_diam', 'std_logA', 'std_logd_int', 'cv_area_int', 'max_over_mean_d', 'q90_q10_d', 'gini_area', 'cv_diam_aw', 'std_d_over_mean_aw', 'mean_absdev_logA']
out = np.full((500, len(defs)), np.nan)
for n in range(500):
    if nz[n] >= 11.9: continue
    m = masks[mids.index(ids[n])].astype(np.int64); labs = np.unique(m); labs = labs[labs > 0]
    if len(labs) < 15: continue
    a = ndi.sum(np.ones(m.shape), m, labs)
    border = np.isin(labs, np.unique(np.r_[m[0], m[-1], m[:, 0], m[:, -1]]))
    ai = a[~border] if (~border).sum() >= 10 else a
    d = np.sqrt(4 * a / np.pi); di = np.sqrt(4 * ai / np.pi)
    srt = np.sort(a); cum = np.cumsum(srt); gini = 1 - 2 * np.sum(cum) / (len(srt) * cum[-1]) + 1 / len(srt)
    w = a / a.sum(); mdw = (w * d).sum()
    out[n] = [a.std() / a.mean(), d.std() / d.mean(), np.log(a).std(), np.log(di).std(), ai.std() / ai.mean(), d.max() / d.mean(), np.quantile(d, .9) / np.quantile(d, .1), gini,
              np.sqrt((w * (d - mdw) ** 2).sum()) / mdw, np.sqrt((w * (d - mdw) ** 2).sum()) / d.mean(), np.mean(np.abs(np.log(a) - np.log(a).mean()))]
s = ~np.isnan(out[:, 0])
ctrl = ['u_ph', 'z_f25', 'z_f50', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac']
C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
res = []
for j, k in enumerate(defs + ['het4_u', 'sizefield_std_s32', 'm_la_std', 'grain_logA_std', 'tile_logA_std', 'het4_u_mx']):
    v = out[s, j] if j < len(defs) else X[s, keys.index(k)]
    rv = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]; res.append((np.corrcoef(rv, ry)[0, 1], k))
for c, k in sorted(res, key=lambda t: -abs(t[0])): print('%-20s %+.3f' % (k, c))
np.savez(_repo_io('hetdef-train.npz'), X=out, keys=np.array(defs), ids=np.array(ids))
