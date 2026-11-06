"""Probe: sampling-bias-corrected block heterogeneity of log grain size from U-Net maps (z1), label-free."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv, sys
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X'])
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
res = np.zeros((500, 6))
for n in range(500):
    m = maps[n].astype(np.float64); g = m[0] + m[1]; la = m[4]; a = np.exp(la)
    for bi, K in enumerate((4, 8)):
        s = 128 // K; bm = []; bv = []
        for i in range(K):
            for j in range(K):
                sl = (slice(i * s, (i + 1) * s), slice(j * s, (j + 1) * s)); w = g[sl]; W = w.sum()
                if W < 0.3 * s * s: continue
                mu = (w * la[sl]).sum() / W; var_px = (w * (la[sl] - mu) ** 2).sum() / W
                neff = (W * 4) / max((w * a[sl]).sum() / W, 1.0)          # grains in block (full-res px = 4 * map px)
                bm.append(mu); bv.append(var_px / max(neff, 1.0))
        bm = np.array(bm); bv = np.array(bv); raw = bm.var(); bias = bv.mean()
        res[n, bi * 3:(bi + 1) * 3] = [np.sqrt(raw), np.sqrt(max(raw - bias, 0)), raw - bias]
names = ['het4_raw', 'het4_db', 'het4_db2', 'het8_raw', 'het8_db', 'het8_db2']
np.save(_repo_io('hetdb-train.npy'), res)
nz = X[:, keys.index('noise')] * 255; nobj = X[:, keys.index('n_obj')]
ctrl = ['u_ph', 'z_f25', 'z_f50', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac']
for nm, s in [('HQ', nz < 11.9), ('HQ N<250', (nz < 11.9) & (nobj < 250)), ('HQ N>=250', (nz < 11.9) & (nobj >= 250)), ('all', nz > -1)]:
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    line = []
    for j, k in enumerate(names):
        v = res[s, j]; rv = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]; line.append('%s %+.3f' % (k, np.corrcoef(rv, ry)[0, 1]))
    for k in ['het4_u', 'sizefield_std_s32', 'm_la_std']:
        v = X[s, keys.index(k)]; rv = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]; line.append('%s %+.3f' % (k, np.corrcoef(rv, ry)[0, 1]))
    print(nm, s.sum(), ' | '.join(line))
