"""053 structure with the Ridge / log-Ridge components replaced on noise-gated rows by alternative OOFs (sr-*.npz from synaug_ridge2.py).
Gate is a-priori (pixel noise band), nothing fitted. Also reports a same-features no-sim control (sr-H2Z-none) to separate the feature-set
difference (H2Z vs production anghcor) from the effect of the degraded copies."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
B = load(3); y = B['y']; fail = B['fail']
D = np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True); nz = np.nan_to_num(D['noise'], nan=0)
def c053(ridge, logr):
    c = 0.16 * B['p021'] + 0.384 * B['g'] + 0.128 * ridge + 0.128 * logr + 0.05 * B['set'] + 0.15 * B['gmm']; return np.where(fail, B['p021'], c)
def rep(a, tag, base):
    e = (a - y) ** 2; eb = (base - y) ** 2
    bands = {'clean': nz < 9.5, 'mid': (nz >= 9.5) & (nz < 12), 'noisy': nz >= 12}
    print(f'{tag:34s} {np.sqrt(e.mean()):.4f} (gain {np.sqrt(eb.mean()) - np.sqrt(e.mean()):+.4f}; parts ' + '/'.join(f'{np.sqrt(eb[i].mean()) - np.sqrt(e[i].mean()):+.3f}' for i in range(3)) + ')  ' +
          ' '.join(f'{b} {np.sqrt(e[:, m].mean()):.3f}' for b, m in bands.items()))
base = c053(B['ridge'], B['logridge']); rep(base, '053', base)
for f in sys.argv[1:]:
    S = np.load(_repo_io(f'{f}.npz'))
    for thr in (12.0, 9.5):
        g = nz >= thr
        r = np.where(g, S['ridge'], B['ridge']); l = np.where(g, S['logridge'], B['logridge']); rep(c053(r, l), f'{f} on noise>={thr}', base)
    rep(c053(S['ridge'], S['logridge']), f'{f} on all rows', base)
