"""Does each 64-px column/row of the image contribute equally? y ~ controls + phase fraction in strip k + phase fraction elsewhere; ratio of per-area coefficients."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X'])
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
PH = np.zeros((500, 2, 4)); G = np.zeros((500, 2, 4))      # [axis(col,row), strip] phase-prob sum and grain-prob sum
for n in range(500):
    m = maps[n].astype(np.float64); g = m[0] + m[1]; p = m[1]
    for k in range(4):
        PH[n, 0, k] = p[:, 32 * k:32 * (k + 1)].sum(); G[n, 0, k] = g[:, 32 * k:32 * (k + 1)].sum()
        PH[n, 1, k] = p[32 * k:32 * (k + 1), :].sum(); G[n, 1, k] = g[32 * k:32 * (k + 1), :].sum()
nz = X[:, keys.index('noise')] * 255
ctrl = ['het4_u', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac', 'z_f50']
rng = np.random.RandomState(0)
for nm, s in [('LQ', nz >= 11.9), ('HQ', nz < 11.9)]:
    idx = np.where(s)[0]
    for ax, axn in [(0, 'col'), (1, 'row')]:
        line = []
        for k in range(4):
            # strip phase "area contribution" as fraction of whole-image grain area: strip and rest
            tot = G[:, ax].sum(1); fk = PH[:, ax, k] / tot; fr = (PH[:, ax].sum(1) - PH[:, ax, k]) / tot
            def ratio(ii):
                C = np.c_[np.ones(len(ii)), X[ii][:, [keys.index(c) for c in ctrl]], A[ii, :2], fk[ii], fr[ii]]
                c = np.linalg.lstsq(C, y[ii], rcond=None)[0]; return c[-2] / c[-1]
            r0 = ratio(idx); B = [ratio(rng.choice(idx, len(idx))) for _ in range(300)]
            line.append('%s%d coef ratio strip/rest %.2f [%.2f, %.2f]' % (axn, k, r0, np.percentile(B, 5), np.percentile(B, 95)))
        print(nm, ' | '.join(line))
