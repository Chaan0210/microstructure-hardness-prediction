"""Spatially weighted phase fraction: w = Gaussian(x0,y0,sigma) over the image; scan on HQ, validate on LQ (independent rows)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv, itertools
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X'])
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
P = np.stack([maps[n, 1].astype(np.float32) for n in range(500)]); Gm = np.stack([(maps[n, 0] + maps[n, 1]).astype(np.float32) for n in range(500)])
nz = X[:, keys.index('noise')] * 255; hq = nz < 11.9; lq = ~hq
ctrl = ['het4_u', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac', 'z_f50']
def pc(v, s):
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(c) for c in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    rv = v[s] - C @ np.linalg.lstsq(C, v[s], rcond=None)[0]; return np.corrcoef(rv, ry)[0, 1]
yy, xx = np.mgrid[0:128, 0:128] * 2 + 1.0
uni = (P.sum((1, 2)) / Gm.sum((1, 2)))
print('uniform phase: HQ %.3f LQ %.3f all %.3f' % (pc(uni, hq), pc(uni, lq), pc(uni, hq | lq)))
res = []
for x0, y0, sg in itertools.product(range(16, 256, 32), range(16, 256, 32), (48, 64, 96, 128, 192)):
    w = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sg ** 2)).astype(np.float32)
    v = (P * w).sum((1, 2)) / (Gm * w).sum((1, 2))
    res.append((pc(v, hq), x0, y0, sg, pc(v, lq)))
res.sort(reverse=True)
for r in res[:10]: print('HQ %.3f  x0 %3d y0 %3d sigma %3d | LQ (independent) %.3f' % r)
# x-only weighting (vertical strips): gaussian in x only
res2 = []
for x0, sg in itertools.product(range(0, 257, 16), (32, 48, 64, 96, 128, 192)):
    w = np.exp(-((xx - x0) ** 2) / (2 * sg ** 2)).astype(np.float32); v = (P * w).sum((1, 2)) / (Gm * w).sum((1, 2)); res2.append((pc(v, hq), x0, sg, pc(v, lq)))
res2.sort(reverse=True)
for r in res2[:6]: print('x-only HQ %.3f x0 %3d sigma %3d | LQ %.3f' % r)
