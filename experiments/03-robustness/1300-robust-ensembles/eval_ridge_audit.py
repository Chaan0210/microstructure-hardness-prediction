from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
A = np.load(_repo_io('audit061.npz')); y, nz, fail, c61 = A['y'], A['nz'], A['fail'], A['c61']; C = {k[2:]: A[k] for k in A.files if k.startswith('C_')}
R = np.load(_repo_io('ridge_audit.npz')); g = (nz >= 12) & ~fail[0]
import sys; sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load; B = load(3)
W = dict(cnn=0.16, tab=0.384, ridge=0.128, logr=0.128, set=0.05, gmm=0.15)
r = lambda a, m=slice(None): float(np.sqrt(((a - y)[:, m] ** 2).mean()))
pr = lambda a: '/'.join(f'{u:+.3f}' for u in np.sqrt(((c61 - y) ** 2).mean(1)) - np.sqrt(((a - y) ** 2).mean(1)))
for nm in ('base', 'eqw', 'grp', 'eqw+grp'):
    Cx = dict(C); Cx['ridge'] = np.where(g, R[f'{nm}_ridge'], B['ridge']); Cx['logr'] = np.where(g, R[f'{nm}_logridge'], B['logridge'])
    a = np.where(fail, C['cnn'], sum(W[k] * Cx[k] for k in W))
    e = (a - y) ** 2 - (c61 - y) ** 2; se = e[:, g].mean(0).std() / np.sqrt(g.sum())
    print(f'{nm:8s} dev {r(a):.4f} ({r(c61) - r(a):+.4f}; parts {pr(a)}) noisy {r(a, g):.3f} dMSE_noisy {e[:, g].mean():+.2f} (SE {se:.2f}) | ridge alone {r(R[nm + "_ridge"], g):.3f} logr alone {r(R[nm + "_logridge"], g):.3f}')
