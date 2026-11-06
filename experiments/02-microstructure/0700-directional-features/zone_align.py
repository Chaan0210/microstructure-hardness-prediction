"""Zone hypothesis test: 4x4 block heterogeneity with aligned vs shifted grids; number-density estimators."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X'])
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
def blocks(n_off, s=32, K=4):
    # returns list of (slice,slice) for a KxK grid of block size s (map px) offset by n_off map px, cropped to the image
    out = []
    for a in range(-1, K + 1):
        for b in range(-1, K + 1):
            y0, x0 = a * s + n_off, b * s + n_off
            y1, x1 = y0 + s, x0 + s
            y0c, x0c, y1c, x1c = max(y0, 0), max(x0, 0), min(y1, 128), min(x1, 128)
            if (y1c - y0c) * (x1c - x0c) >= 0.75 * s * s: out.append((slice(y0c, y1c), slice(x0c, x1c)))
    return out
offs = [0, 8, 16, 24]
res = np.zeros((500, len(offs), 3))
for n in range(500):
    m = maps[n].astype(np.float64); g = m[0] + m[1]; la = m[4]; a = np.exp(la); e = np.exp(-0.5 * la)
    for t, off in enumerate(offs):
        hp = []; lam = []; dens = []
        for sl in blocks(off):
            W = g[sl].sum()
            if W < 0.2 * g[sl].size: continue
            hp.append(-2 * np.log((g[sl] * e[sl]).sum() / W)); lam.append((g[sl] * la[sl]).sum() / W); dens.append(np.log((g[sl] / a[sl]).sum() * 4 / W + 1e-9))
        res[n, t] = [np.std(hp), np.std(lam), np.std(dens)]
nz = X[:, keys.index('noise')] * 255
ctrl = ['u_ph', 'z_f25', 'z_f50', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac']
for nm, s in [('HQ', nz < 11.9), ('LQ', nz >= 11.9), ('all', nz > -1)]:
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    line = []
    for t, off in enumerate(offs):
        line.append('off%2dpx: ' % (2 * off) + ' '.join('%s %+.3f' % (k, np.corrcoef(res[s, t, q] - C @ np.linalg.lstsq(C, res[s, t, q], rcond=None)[0], ry)[0, 1]) for q, k in enumerate(['hp', 'la', 'dens'])))
    print(nm, ' || '.join(line))
np.save(_repo_io('zone-train.npy'), res)
