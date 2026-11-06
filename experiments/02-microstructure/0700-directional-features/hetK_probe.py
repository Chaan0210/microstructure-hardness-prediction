from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X'])
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
Ks = [2, 3, 4, 5, 6, 8, 16]
out = np.zeros((500, len(Ks) * 2))
for n in range(500):
    m = maps[n].astype(np.float64); g = m[0] + m[1]; e = np.exp(-0.5 * m[4])
    for j, K in enumerate(Ks):
        edges = np.linspace(0, 128, K + 1).astype(int); b = []; bl = []
        for i0 in range(K):
            for j0 in range(K):
                sl = (slice(edges[i0], edges[i0 + 1]), slice(edges[j0], edges[j0 + 1])); W = g[sl].sum()
                if W < 0.2 * g[sl].size: continue
                b.append(-2 * np.log((g[sl] * e[sl]).sum() / W)); bl.append((g[sl] * m[4][sl]).sum() / W)
        out[n, 2 * j] = np.std(b); out[n, 2 * j + 1] = np.std(bl)
nz = X[:, keys.index('noise')] * 255
ctrl = ['u_ph', 'z_f25', 'z_f50', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac']
for nm, s in [('HQ', nz < 11.9), ('LQ', nz >= 11.9), ('all', nz > -1)]:
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    line = []
    for j, K in enumerate(Ks):
        for t, nm2 in [(0, 'hp'), (1, 'la')]:
            v = out[s, 2 * j + t]; rv = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]; line.append('K%d%s %+.3f' % (K, nm2, np.corrcoef(rv, ry)[0, 1]))
    v = X[s, keys.index('het4_u')]; rv = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]; line.append('het4_u %+.3f' % np.corrcoef(rv, ry)[0, 1])
    print(nm, ' | '.join(line))
np.save(_repo_io('hetK-train.npy'), out)
