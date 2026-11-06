from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys
src = open(_repo_io('holdout_blend.py')).read(); src = src[:src.index("e61 = (P['061'] - y) ** 2")]
exec(compile(src, 'holdout_blend', 'exec'))
cnn = cc; rest61 = (P['061'] - 0.16 / 0.95 * cnn) / (1 - 0.16 / 0.95)
base = P['061']; eb = (base - y) ** 2
for s in [0.0, 0.05, 0.1, 0.168, 0.25]:
    a = s * cnn + (1 - s) * rest61; e = (a - y) ** 2; d = e - eb
    print(f'cnn share {s:.3f}: holdout all {np.sqrt(e.mean()):.3f} (dMSE {d.mean():+.2f} SE {d.std() / np.sqrt(len(d)):.2f}) noisy {np.sqrt(e[g].mean()):.3f} clean {np.sqrt(e[nz < 9.5].mean()):.3f}')
print('--- noisy-only CNN share (clean/mid rows keep 061) ---')
for s in [0.0, 0.05, 0.1]:
    a = np.where(g, s * cnn + (1 - s) * rest61, base); e = (a - y) ** 2; d = (e - eb)[g]
    print(f'noisy cnn share {s}: holdout all {np.sqrt(e.mean()):.3f} noisy {np.sqrt(e[g].mean()):.3f} dMSE_noisy {d.mean():+.2f} SE {d.std() / np.sqrt(g.sum()):.2f}')
