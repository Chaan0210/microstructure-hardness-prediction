"""Learning-curve-extrapolated stacking (train only): from component OOF at training sizes n = 160/240/320 (lc-*.npz, identical subsets),
fit every entry of the error second-moment matrix M_ij(n) = E[e_i e_j] as a + b/n, extrapolate to n = 500 (the size the submitted models are
trained on) and solve the simplex-constrained optimal blend weights at each n. usage: lc_analyze.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy.optimize import minimize
sid_y = np.load(_repo_io('dev065.npz'))['y']; nz = np.load(_repo_io('dev065.npz'))['nz']; fail = np.load(_repo_io('dev065.npz'))['fail']
C = ['cnn', 'tab', 'ridge', 'logridge', 'gmm']; F = [0.5, 0.75, 1.0]; N = np.array([160, 240, 320])
E = {}
for f in F:
    T = np.load(_repo_io(f'lc-tab-f{f:g}.npz')); Cn = np.load(_repo_io(f'lc-cnn-f{f:g}.npz')); E[f] = np.stack([(Cn['cnn'] if c == 'cnn' else T[c]) - sid_y for c in C])[:, :, ~fail]   # comp x part x rows
def M(e, mask=None):
    x = e.reshape(len(C), -1) if mask is None else e[:, :, mask[~fail]].reshape(len(C), -1); return x @ x.T / x.shape[1]
def wopt(Mx):
    k = len(C); r = minimize(lambda w: w @ Mx @ w, np.ones(k) / k, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}], bounds=[(0, 1)] * k, method='SLSQP'); return r.x
w53 = np.array([0.16, 0.384, 0.128, 0.128, 0.15]); w53 = w53 / w53.sum()
for label, mask in (('all rows', None), ('noisy rows', nz >= 12), ('clean+mid', nz < 12)):
    Ms = [M(E[f], mask) for f in F]; A = np.zeros_like(Ms[0]); Bc = np.zeros_like(Ms[0])
    for i in range(len(C)):
        for j in range(len(C)):
            b, a = np.polyfit(1 / N, [m[i, j] for m in Ms], 1); A[i, j], Bc[i, j] = a, b
    M500 = A + Bc / 500
    print(f'== {label}')
    print('   component RMSE n=160/240/320 -> 500(extrap): ' + '  '.join(f'{c} ' + '/'.join(f'{np.sqrt(m[i, i]):.2f}' for m in Ms) + f'->{np.sqrt(M500[i, i]):.2f}' for i, c in enumerate(C)))
    for n, Mx in list(zip(N, Ms)) + [(500, M500)]:
        w = wopt(Mx); print(f'   n={n:3d}: opt w ' + ' '.join(f'{c} {x:.2f}' for c, x in zip(C, w)) + f' | RMSE opt {np.sqrt(w @ Mx @ w):.3f}  053-weights {np.sqrt(w53 @ Mx @ w53):.3f}')
