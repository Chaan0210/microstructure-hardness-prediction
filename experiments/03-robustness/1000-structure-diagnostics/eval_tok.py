"""Evaluate branch-1 variants (var-tok-<v>/seed*/oof.npz) and any extra OOF files: seed-mean standalone RMSE, error correlation with the grain
block and with 053, and as an extra partner on top of 053 at fixed shares (all variants compared at the same shares; struct = control).
usage: eval_tok.py [extra_name=path.npz:key ...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, _repo_io('../push-900'))
from blend_dev import load, rmse, parts
B = load(9); y, fail, p = B['y'], B['fail'], B['p021']; W = lambda a: np.where(fail, p, a); S3 = B['sets'][:3].mean(0)
c53 = W(0.80 * B['c051'] + 0.05 * S3 + 0.15 * B['gmm']); g = W(B['g'])
M = {}
for pre in ('tok', 'tokdino'):
    for v in ('struct', 'generic', 'cond'):
        fs = sorted(Path(f'var-{pre}-{v}').glob('seed*/oof.npz'))
        if fs: M[f'{pre}-{v} ({len(fs)} seeds)'] = np.mean([np.load(_repo_io(f))['prediction'] for f in fs], 0)
for a in sys.argv[1:]:
    nm, rest = a.split('='); path, key = rest.split(':'); M[nm] = np.load(_repo_io(path))[key]
for nm, o in M.items():
    oo = W(o); e = (oo - y).ravel()
    line = f'{nm:26s} alone {rmse(oo, y):.3f}  corr(grain) {np.corrcoef(e, (g - y).ravel())[0, 1]:.3f}  corr(053) {np.corrcoef(e, (c53 - y).ravel())[0, 1]:.3f} |'
    for s in (0.05, 0.1):
        c = c53 + s * (oo - c53); line += f' a{s}: {rmse(c53, y) - rmse(c, y):+.4f} (' + '/'.join(f'{v:+.3f}' for v in parts(c53, y) - parts(c, y)) + ')'
    d = (oo - c53).ravel(); a_opt = -np.mean(d * (c53 - y).ravel()) / np.mean(d * d); print(line + f'  opt a {a_opt:+.3f}')
