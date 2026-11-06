"""Honest per-band simplex stacking (NNLS weights fitted on 4 folds' OOF rows of a partition, scored on the 5th) for the 073 components on dev
(non-gated rows use the nb TabPFN block and nb partners). Structure check: is the fixed weight vector still at the honest optimum?"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy.optimize import nnls
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import align_dev
import nsgain as N
y, g, c61, C, nz, fail, fold, W = N.y, N.g, N.c61, N.C, N.nz, N.fail, N.fold, N.W; f = fail[0]; ng = ~g & ~f
S = ['038', '039', '044']; V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']; Tn = np.mean([align_dev(f'abl/dev-nb-{s}-{v}.npz') for s in S for v in V], 0)
Z = np.load(_repo_io('nb-partners.npz')); Cx = dict(C); Cx['tab'] = np.where(ng, Tn, C['tab'])
for nm, key in (('ridge', 'ridge'), ('logr', 'logridge'), ('gmm', 'gmm')): Cx[nm] = np.where(ng, C[nm] + (Z[key + '_nb_dev'] - Z[key + '_full_dev']), C[nm])
c73 = N.blend(Cx); comps = list(W)
def stack(m):
    out = c73.copy()
    for p in range(3):
        for k in range(5):
            tr = (fold[p] != k) & m & ~f; va = (fold[p] == k) & m
            A = np.array([Cx[c][p, tr] for c in comps]).T; w, _ = nnls(A, y[tr]); w /= w.sum(); out[p, va] = np.array([Cx[c][p, va] for c in comps]).T @ w
    return out
rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
for nm, m in (('clean<11', (nz < 11) & ~f), ('<12', ng), ('noisy>=12', g)):
    s = stack(m); A = np.array([Cx[c][:, m].ravel() for c in comps]).T; w, _ = nnls(A, np.tile(y[m], 3))
    print(f'{nm:9s} 073 {rb(c73, m):.3f} honest stack {rb(s, m):.3f} | in-sample weights ' + ' '.join(f'{c} {x:.2f}' for c, x in zip(comps, w / w.sum())) + ' | fixed ' + ' '.join(f'{c} {W[c]:.2f}' for c in comps))
print('073 dev %.4f (061 %.4f)' % (N.rm(c73), N.rm(c61)))
