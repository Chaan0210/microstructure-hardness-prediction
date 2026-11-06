"""Block-scale scan of spatial variation of phase fraction, darkness z and orientation (U-Net z1 maps) vs hardness."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X'])
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
Ks = [2, 3, 4, 5, 6, 8]
out = np.zeros((500, len(Ks), 4))
for n in range(500):
    m = maps[n].astype(np.float64); g = m[0] + m[1]; zf = g / (1 + np.exp(-(m[7] - 0.25) / 0.02))
    for j, K in enumerate(Ks):
        e = np.linspace(0, 128, K + 1).astype(int); ph = []; zz = []; cc = []; ss = []
        for a in range(K):
            for b in range(K):
                sl = (slice(e[a], e[a + 1]), slice(e[b], e[b + 1])); W = g[sl].sum()
                if W < 0.2 * g[sl].size: continue
                ph.append(m[1][sl].sum() / W); zz.append(zf[sl].sum() / W); cc.append((g[sl] * m[5][sl]).sum() / W); ss.append((g[sl] * m[6][sl]).sum() / W)
        cc = np.array(cc); ss = np.array(ss)
        out[n, j] = [np.std(ph), np.std(zz), np.sqrt(cc.var() + ss.var()), np.hypot(cc.mean(), ss.mean()) / np.mean(np.hypot(cc, ss))]
nz = X[:, keys.index('noise')] * 255
ctrl = ['u_ph', 'z_f25', 'z_f50', 'het4_u', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac']
names = ['ph_std', 'zf_std', 'orient_std', 'align_ratio']
for nm, s in [('HQ', nz < 11.9), ('LQ', nz >= 11.9)]:
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    for t, nm2 in enumerate(names):
        print(nm, '%-11s' % nm2, ' '.join('K%d %+.3f' % (K, np.corrcoef(out[s, j, t] - C @ np.linalg.lstsq(C, out[s, j, t], rcond=None)[0], ry)[0, 1]) for j, K in enumerate(Ks)))
