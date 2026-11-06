"""Blend-level holdout-100 check of DTA partners with the realistic uplift copies (run from push-1200): partners refit on dev-400 + copies of dev
sources only; proxy 061 structure (holdout_blend.py). Variants: 061 (d1234), d1234+degU1(+degU2), degU only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
src = open(_repo_io('holdout_blend.py')).read(); src = src[:src.index("e61 = (P['061'] - y) ** 2")]; exec(compile(src, 'hb', 'exec'))
e61 = (P['061'] - y) ** 2
avail = [s for s in ('degU1', 'degU2') if __import__('os').path.exists(f'synthrows-{s}-u1dmavgH2Z-anghcor.npz')]
for nm, sets in [('d1234+U', 'deg,deg2,deg3,deg4'.split(',') + avail), ('U only', avail)]:
    pr, pl, _, _ = partners(sets, False); pb = blend(cc, sp, np.where(g, pr, r0), np.where(g, pl, l0)); e = (pb - y) ** 2; d = (e - e61)[g]
    print(f'{nm:9s} ({"+".join(avail)}) partners noisy: ridge {np.sqrt(((pr - y)[g] ** 2).mean()):.3f} logr {np.sqrt(((pl - y)[g] ** 2).mean()):.3f} (061: {np.sqrt(((r61 - y)[g] ** 2).mean()):.3f} / {np.sqrt(((l61 - y)[g] ** 2).mean()):.3f}) | blend noisy dMSE vs 061 {d.mean():+.2f} (SE {d.std() / np.sqrt(g.sum()):.2f})')
