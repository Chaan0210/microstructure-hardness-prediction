"""Clean zone-alignment test: always exactly 3x3 fully-interior 64-px blocks (same count, size, area); only the grid offset o changes.
o in {0, 64}: block edges on multiples of 64 (aligned); o = 32: maximally misaligned; o in {16, 48}: partial. Block stat: pixel-weighted mean log area (U-Net z1)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X'])
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
offs = [0, 8, 16, 24, 32, 40, 48, 56, 64]          # image px (map px = /2)
H = np.zeros((500, len(offs)))
for n in range(500):
    m = maps[n].astype(np.float64); g = m[0] + m[1]; la = m[4]
    for t, o in enumerate(offs):
        om = o // 2; b = []
        for a_ in range(3):
            for c_ in range(3):
                sl = (slice(om + 32 * a_, om + 32 * (a_ + 1)), slice(om + 32 * c_, om + 32 * (c_ + 1))); W = g[sl].sum(); b.append((g[sl] * la[sl]).sum() / W)
        H[n, t] = np.std(b)
nz = X[:, keys.index('noise')] * 255
ctrl = ['u_ph', 'z_f25', 'z_f50', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac']
for nm, s in [('HQ', nz < 11.9), ('LQ', nz >= 11.9), ('all', nz > -1)]:
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    print(nm, ' '.join('o%2d %+.3f' % (o, np.corrcoef(H[s, t] - C @ np.linalg.lstsq(C, H[s, t], rcond=None)[0], ry)[0, 1]) for t, o in enumerate(offs)))
