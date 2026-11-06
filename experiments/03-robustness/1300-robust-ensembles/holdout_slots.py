from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
src = open(_repo_io('holdout_blend.py')).read(); src = src[:src.index("e61 = (P['061'] - y) ** 2")]; exec(compile(src, 'hb', 'exec'))
for s in ('038', '039', '044'): print(f'slot {s} holdout all {np.sqrt(((T[f"{s}_plain"] - y) ** 2).mean()):.3f} clean+mid {np.sqrt(((T[f"{s}_plain"] - y)[~g] ** 2).mean()):.3f} noisy {np.sqrt(((T[f"{s}_plain"] - y)[g] ** 2).mean()):.3f}')
e61 = (P['061'] - y) ** 2
for nm, w in [('equal (061)', (1/3, 1/3, 1/3)), ('044 only', (0, 0, 1)), ('dev in-fold mean (.02,.05,.93)', (0.019, 0.05, 0.931)), ('039+044', (0, .5, .5))]:
    blk = sum(wi * T[f'{s}_plain'] for wi, s in zip(w, ('038', '039', '044')))
    pr = blend(cc, blk, np.where(g, r61, r0), np.where(g, l61, l0)); e = (pr - y) ** 2; d = e - e61
    print(f'{nm:32s} holdout blend {np.sqrt(e.mean()):.3f} dMSE {d.mean():+.2f} (SE {d.std() / np.sqrt(len(d)):.2f}) clean+mid {d[~g].mean():+.2f} noisy {d[g].mean():+.2f}')
