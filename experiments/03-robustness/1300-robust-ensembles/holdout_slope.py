from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
src = open(_repo_io('holdout_blend.py')).read(); src = src[:src.index("e61 = (P['061'] - y) ** 2")]; exec(compile(src, 'hb', 'exec'))
def slope(pred, yy):
    x = pred - pred.mean(); b = (x * (yy - yy.mean())).sum() / (x * x).sum(); res = yy - yy.mean() - b * x
    return b, np.sqrt((res ** 2).sum() / (len(x) - 2) / (x * x).sum())
for nm, m in [('clean', nz < 9.5), ('mid', (nz >= 9.5) & (nz < 12)), ('noisy', g)]:
    b, s = slope(P['061'][m], y[m]); print(f'holdout {nm}: n {m.sum()} slope {b:.3f}±{s:.3f}  y sd {y[m].std():.1f} pred sd {P["061"][m].std():.1f}')
