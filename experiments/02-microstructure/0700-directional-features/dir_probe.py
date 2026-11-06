"""Probe: directional (non-rotation-invariant) image properties vs hardness (partial corr, D4-symmetric U-Net controls + angle)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X']); assert [str(i) for i in F['ids']] == ids
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
yy, xx = np.mgrid[0:128, 0:128] / 127.0 - 0.5
D = []
for n in range(len(ids)):
    m = maps[n].astype(np.float32); g = m[0] + m[1]; la = m[4]; ph = m[1] / np.maximum(g, 1e-3); z = m[7]
    row = []
    for fld, w in [(la, g), (ph, g), (m[2], np.ones_like(g)), (z, g)]:   # size field, phase field, pore field, darkness
        W_ = w / w.sum(); Bm = np.c_[np.ones(128 * 128), xx.ravel(), yy.ravel()] * np.sqrt(W_.ravel())[:, None]
        c = np.linalg.lstsq(Bm, fld.ravel() * np.sqrt(W_.ravel()), rcond=None)[0]; row += [c[1], c[2]]   # x / y gradient
    # centroid of phase / pores relative to center
    D.append(row)
D = np.array(D); names = ['la_gx', 'la_gy', 'ph_gx', 'ph_gy', 'pore_gx', 'pore_gy', 'z_gx', 'z_gy']
nz = X[:, keys.index('noise')] * 255
ctrl = ['u_ph', 'z_f25', 'z_f50', 'het4_u', 'het4_u_mx', 'm_la_std', 'm_la_mean', 'm_la_f32', 'm_el_pix', 'm_el_glob', 'n_obj', 'sizefield_std_s32']
for nm, s in [('all', nz > -1), ('HQ', nz < 11.9)]:
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    out = []
    for j, k in enumerate(names):
        for tr, v in [('', D[:, j]), ('|.|', np.abs(D[:, j]))]:
            rv = v[s] - C @ np.linalg.lstsq(C, v[s], rcond=None)[0]; out.append('%s%s %+.3f' % (tr, k, np.corrcoef(rv, ry)[0, 1]))
    # gradient magnitude and direction harmonics for size field
    mag = np.hypot(D[:, 0], D[:, 1]); th = np.arctan2(D[:, 1], D[:, 0])
    for tr, v in [('la_gmag', mag), ('la_c1', mag * np.cos(th)), ('la_s1', mag * np.sin(th)), ('la_c2', mag * np.cos(2 * th)), ('la_s2', mag * np.sin(2 * th)), ('la_c4', mag * np.cos(4 * th))]:
        rv = v[s] - C @ np.linalg.lstsq(C, v[s], rcond=None)[0]; out.append('%s %+.3f' % (tr, np.corrcoef(rv, ry)[0, 1]))
    print(nm, ' | '.join(out))
np.savez(_repo_io('dirgrad-train.npz'), X=D, keys=np.array(names), ids=np.array(ids))
for nm, s in [('LQ', nz >= 11.9), ('HQ el<0.5', (nz < 11.9) & (X[:, keys.index('m_el_glob')] < 0.5)), ('HQ el>=0.5', (nz < 11.9) & (X[:, keys.index('m_el_glob')] >= 0.5))]:
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    print(nm, s.sum(), ' | '.join('%s %+.3f' % (k, np.corrcoef(D[s, j] - C @ np.linalg.lstsq(C, D[s, j], rcond=None)[0], ry)[0, 1]) for j, k in enumerate(names)))
