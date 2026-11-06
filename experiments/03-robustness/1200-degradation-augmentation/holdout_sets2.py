"""Holdout-100: which added copy set changes the partners? d1234 (+ deg5 | deg6 | deg7 | deg8 | deg9 | deg10), production alphas; paired SE over the 35 noisy rows.
usage: holdout_sets2.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
exec(open(_repo_io('holdout_sets.py')).read().split('for name, sets in')[0].split('"""', 2)[2])
def mean_pred(sets):
    pr, y, nz = run('u1dmavgH2Z', sets, False, 'ridge'); pl = np.exp(np.mean([np.log(run(v, sets, False, 'log')[0]) for v in V], 0)); return (pr + pl) / 2, y, nz >= 12
b, y, g = mean_pred('deg,deg2,deg3,deg4'.split(',')); eb = (b - y)[g]
print(f'd1234 base noisy {np.sqrt((eb ** 2).mean()):.3f}')
for add in ['deg5', 'deg6', 'deg7', 'deg8', 'deg9', 'deg10', 'deg5,deg6,deg7']:
    p, _, _ = mean_pred('deg,deg2,deg3,deg4'.split(',') + add.split(',')); e = (p - y)[g]; d = e ** 2 - eb ** 2
    print(f'+{add:15s} noisy {np.sqrt((e ** 2).mean()):.3f}  dMSE {d.mean():+.2f} (SE {d.std() / np.sqrt(len(d)):.2f})  mean shift {(p - b)[g].mean():+.2f}  sd ratio {p[g].std() / b[g].std():.3f}')
