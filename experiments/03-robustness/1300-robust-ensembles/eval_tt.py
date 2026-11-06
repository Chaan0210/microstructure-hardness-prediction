"""Evaluate TT-DTA students (tt-<tag>.npz) on the official dev: standalone noisy-row RMSE, error correlation with 061 components,
061 with the student replacing a component on noisy rows at the same weight, and fold-honest noisy-band simplex stack including the student.
usage: eval_tt.py TAG [TAG ...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy.optimize import nnls
A = np.load(_repo_io('audit061.npz')); y, nz, fail, fold, c61 = A['y'], A['nz'], A['fail'], A['fold'], A['c61']
C = {k[2:]: A[k] for k in A.files if k.startswith('C_')}
W = dict(cnn=0.16, tab=0.384, ridge=0.128, logr=0.128, set=0.05, gmm=0.15)
g = (nz >= 12) & ~fail[0]
r = lambda a, m=slice(None): float(np.sqrt(((a - y)[:, m] ** 2).mean()))
def blend(Cx): return np.where(fail, C['cnn'], sum(W[k] * Cx[k] for k in W))
def report(nm, a): print(f'  {nm:42s} dev {r(a):.4f} ({r(c61) - r(a):+.4f}; parts ' + '/'.join(f'{u:+.3f}' for u in np.sqrt(((c61 - y) ** 2).mean(1)) - np.sqrt(((a - y) ** 2).mean(1))) + f') noisy {r(a, g):.3f}')
print(f'061 dev {r(c61):.4f} noisy {r(c61, g):.3f}; components noisy: ' + ' '.join(f'{k} {r(v, g):.2f}' for k, v in C.items()))
for tag in sys.argv[1:]:
    T = np.load(_repo_io(f'tt-{tag}.npz'))
    for s in [k for k in T.files if k != 'ids']:
        m = T[s]; e = (m - y)[:, g].ravel()
        cor = {k: np.corrcoef(e, (C[k] - y)[:, g].ravel())[0, 1] for k in ('tab', 'logr', 'ridge')}
        print(f'[{tag}:{s}] noisy alone {r(m, g):.3f} parts ' + '/'.join(f'{np.sqrt(((m - y)[p, g] ** 2).mean()):.2f}' for p in range(3)) +
              f' | all-rows {r(m):.3f} | err corr tab {cor["tab"]:.2f} logr {cor["logr"]:.2f} ridge {cor["ridge"]:.2f}')
        for comp in ('tab', 'logr', 'ridge'):
            Cx = dict(C); Cx[comp] = np.where(g, m, C[comp]); report(f'replace {comp} on noisy rows', blend(Cx))
        for w in (0.1, 0.2, 0.3):
            report(f'add at share {w} on noisy rows', np.where(g & ~fail, (1 - w) * c61 + w * m, c61))
        # fold-honest noisy-band stack incl. student
        comps = list(C) + ['stu']; Cs = dict(C, stu=m); out = c61.copy()
        for p in range(3):
            for k in range(5):
                tr = (fold[p] != k) & g; va = (fold[p] == k) & g
                w, _ = nnls(np.array([Cs[c][p, tr] for c in comps]).T, y[tr]); w /= w.sum(); out[p, va] = np.array([Cs[c][p, va] for c in comps]).T @ w
        w, _ = nnls(np.array([Cs[c][:, g].ravel() for c in comps]).T, np.tile(y[g], 3)); report('honest noisy stack (in-sample w ' + ','.join(f'{c}:{x:.2f}' for c, x in zip(comps, w / w.sum()) if x > 0.005) + ')', out)
