"""Is hardness a NON-uniform spatial average? CV comparison: y ~ global controls + (A) global phase fraction  vs  (B) 16 zone phase fractions (ridge, learned zone weights)
vs (C) 16 zone fractions randomly permuted per image (same marginals, positions destroyed: control). Also zone-level logA / het analog."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
ZV = np.load(_repo_io('zonevec-train.npz'), allow_pickle=True); ids = [str(i) for i in ZV['ids']]; Z = ZV['Z'].astype(float)
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']
PH = np.zeros((500, 16))
for n in range(500):
    m = maps[n].astype(np.float64); g = m[0] + m[1]; PH[n] = (m[1].reshape(4, 32, 4, 32).sum((1, 3)) / g.reshape(4, 32, 4, 32).sum((1, 3))).ravel()
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X']); assert [str(i) for i in F['ids']] == ids
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
nz = X[:, keys.index('noise')] * 255
ctrl = np.c_[X[:, [keys.index(k) for k in ['het4_u', 'het4_u_mx', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac', 'z_f50']]], A[:, :2]]
rng = np.random.RandomState(3); PHs = np.array([PH[n, rng.permutation(16)] for n in range(500)]); ZF = Z[:, :, 4]; ZFs = np.array([ZF[n, rng.permutation(16)] for n in range(500)])
def cv(D, s, lam, reps=20):
    D = D[s]; yy = y[s]; e = []
    for r in range(reps):
        f = np.arange(len(yy)) % 5; np.random.RandomState(r).shuffle(f)
        for k in range(5):
            tr = f != k; mu, sd = D[tr].mean(0), D[tr].std(0) + 1e-9; A_ = (D[tr] - mu) / sd; B_ = (D[~tr] - mu) / sd
            c = np.linalg.solve(A_.T @ A_ + lam * np.eye(A_.shape[1]), A_.T @ (yy[tr] - yy[tr].mean())); e.append(np.mean((B_ @ c + yy[tr].mean() - yy[~tr]) ** 2))
    return np.sqrt(np.mean(e))
for nm, s in [('HQ', nz < 11.9), ('LQ', nz >= 11.9), ('all', nz > -1)]:
    out = []
    for lam in (1, 10, 30):
        a = cv(np.c_[ctrl, PH.mean(1), ZF.mean(1)], s, lam); b = cv(np.c_[ctrl, PH, ZF], s, lam); c = cv(np.c_[ctrl, PHs, ZFs], s, lam)
        out.append('lam%d: uniform %.3f | zone-weighted %.3f | permuted %.3f' % (lam, a, b, c))
    print(nm, s.sum(), ' || '.join(out))
