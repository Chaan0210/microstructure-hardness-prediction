"""Probe: second-phase banding (anisotropy of the smoothed phase-probability field) vs hardness."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
from scipy import ndimage as ndi
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X'])
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
names = []; rows = []
for n in range(500):
    m = maps[n].astype(np.float32); g = m[0] + m[1]; ph = m[1]; la = m[4]
    r = []
    for fld_name, fld in [('ph', ph), ('la', la * g)]:
        for sg in (3, 6, 12):
            f = ndi.gaussian_filter(fld, sg); gx = ndi.sobel(f, 1); gy = ndi.sobel(f, 0)
            jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
            coh = np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-12); th = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
            amp = f.std()
            r += [coh, amp, coh * np.cos(4 * th), coh * np.sin(4 * th)]
            if n == 0: names += [f'{fld_name}_s{sg}_coh', f'{fld_name}_s{sg}_amp', f'{fld_name}_s{sg}_c4', f'{fld_name}_s{sg}_s4']
    rows.append(r)
D = np.array(rows)
nz = X[:, keys.index('noise')] * 255
ctrl = ['u_ph', 'z_f25', 'z_f50', 'het4_u', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'sizefield_std_s32', 'u_ph_f32', 'sp_field_std_s32', 'tile_spfrac_std']
for nm, s in [('all', nz > -1), ('HQ', nz < 11.9), ('LQ', nz >= 11.9)]:
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    print(nm, ' | '.join('%s %+.2f' % (k, np.corrcoef(D[s, j] - C @ np.linalg.lstsq(C, D[s, j], rcond=None)[0], ry)[0, 1]) for j, k in enumerate(names)))
