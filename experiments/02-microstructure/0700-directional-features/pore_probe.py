"""Probe: pore orientation / elongation (from U-Net pore channel, z1 maps) and phase-grain orientation vs hardness."""
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
out = np.zeros((500, 6))
for n in range(500):
    m = maps[n].astype(np.float32); pore = m[2] > 0.5
    lab_, k = ndi.label(pore); C2 = S2 = W = 0.0; ecc = []
    for j in range(1, k + 1):
        yy, xx = np.nonzero(lab_ == j)
        if len(yy) < 4: continue
        cy, cx = yy.mean(), xx.mean(); a = ((xx - cx) ** 2).mean(); b = ((yy - cy) ** 2).mean(); c = ((xx - cx) * (yy - cy)).mean()
        e = np.hypot(a - b, 2 * c) / (a + b + 1e-9); th = 0.5 * np.arctan2(2 * c, a - b)
        C2 += len(yy) * e * np.cos(2 * th); S2 += len(yy) * e * np.sin(2 * th); W += len(yy); ecc.append(e)
    if W > 0:
        C2 /= W; S2 /= W; v = np.hypot(C2, S2) + 1e-9
        out[n] = [np.mean(ecc), v, (C2 * C2 - S2 * S2) / v, 2 * C2 * S2 / v, C2, S2]
    # phase-grain orientation: shape tensor weighted by phase prob
    g = m[1]; c2, s2 = m[5], m[6]
names = ['pore_ecc', 'pore_align', 'pore_c4', 'pore_s4', 'pore_c2', 'pore_s2']
np.savez(_repo_io('pore-orient-train.npz'), X=out, keys=np.array(names), ids=np.array(ids))
nz = X[:, keys.index('noise')] * 255
ctrl = ['u_ph', 'z_f25', 'z_f50', 'het4_u', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'sizefield_std_s32', 'pore_frac', 'px_pore_frac']
for nm, s in [('all', nz > -1), ('HQ', nz < 11.9), ('LQ', nz >= 11.9)]:
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    print(nm, ' | '.join('%s %+.3f' % (k, np.corrcoef(out[s, j] - C @ np.linalg.lstsq(C, out[s, j], rcond=None)[0], ry)[0, 1]) for j, k in enumerate(names)))
print('corr pore_c4 vs grain ang_c4', np.corrcoef(out[:, 2], A[:, 0])[0, 1], 'pore_c2 vs grain C', np.corrcoef(out[:, 4], A[:, 0])[0, 1])
