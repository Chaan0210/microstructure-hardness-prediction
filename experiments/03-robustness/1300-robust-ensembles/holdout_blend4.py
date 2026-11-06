"""Blend-level holdout (push-1200/holdout_blend.py proxy: all components refit on dev-400, TinyCNN stand-in for 021, set NN omitted) for
061 + noisy-row TabPFN slot replacement by TA4 / RS4 (d1234 copies), next to 061 + TA (d1to7, = 067-type). Run from push-1200."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
src = open(_repo_io('holdout_blend.py')).read(); src = src[:src.index("e61 = (P['061'] - y) ** 2")]; exec(compile(src, 'hb', 'exec'))
T4 = np.load(_repo_io('../push-1300/holdout-ta4.npz'), allow_pickle=True); assert [str(i) for i in T4['ids']] == hid
for tag in ('TA4', 'RS4'):
    s4 = np.mean([T4[f'{s}_{tag}'] for s in ('038', '039', '044')], 0)
    P[f'061+{tag}'] = blend(cc, np.where(g, s4, sp), np.where(g, r61, r0), np.where(g, l61, l0))
    print(f'slots {tag}: holdout noisy {np.sqrt(((s4 - y)[g] ** 2).mean()):.3f} (plain {np.sqrt(((sp - y)[g] ** 2).mean()):.3f}, TA-d1to7 {np.sqrt(((sta - y)[g] ** 2).mean()):.3f})')
e61 = (P['061'] - y) ** 2
for k in ('053', '061', '061+TA', '061+TA4', '061+RS4'):
    e = (P[k] - y) ** 2; d = (e - e61)[g]; print(f'{k:8s} all {np.sqrt(e.mean()):.3f} noisy {np.sqrt(e[g].mean()):.3f} | vs 061 noisy dMSE {d.mean():+.2f} (SE {d.std() / np.sqrt(g.sum()):.2f})')
print('--- weight-shift controls (partner share xK on noisy rows, no new model) ---')
base = P['061']; part = (np.where(g, r61, r0) + np.where(g, l61, l0)) / 2; aw = 0.256 / 0.95; other = (base - aw * part) / (1 - aw)
for K in (1.5, 2, 3):
    a = np.where(g, (1 - aw * K) * other + aw * K * part, base); e = (a - y) ** 2; d = (e - e61)[g]
    print(f'partner x{K}: noisy {np.sqrt(e[g].mean()):.3f} dMSE {d.mean():+.2f} (SE {d.std() / np.sqrt(g.sum()):.2f})')
s4 = np.mean([T4[f'{s}_RS4'] for s in ('038', '039', '044')], 0); pl = (l61 + r61) / 2
x = (s4 - sp)[g]; u = (pl - sp)[g]; print('RS4 slot change vs (DTA linear - slot) on holdout noisy: corr %.2f, coef %.2f' % (np.corrcoef(x, u)[0, 1], (x * u).sum() / (u * u).sum()))
