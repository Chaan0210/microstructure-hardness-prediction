from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
src = open(_repo_io('holdout_blend.py')).read(); src = src[:src.index("e61 = (P['061'] - y) ** 2")]; exec(compile(src, 'hb', 'exec'))
e61 = (P['061'] - y) ** 2
for thr in (11.0, 10.4):
    gg = nz >= thr; pb = blend(cc, sp, np.where(gg, r61, r0), np.where(gg, l61, l0)); e = (pb - y) ** 2; band = gg & (nz < 12); d = (e - e61)[band]
    print(f'holdout gate >= {thr}: newly gated rows {band.sum()} dMSE {d.mean():+.2f} (SE {d.std() / np.sqrt(max(band.sum(), 1)):.2f}); all {np.sqrt(e.mean()):.3f} vs 061 {np.sqrt(e61.mean()):.3f}')
