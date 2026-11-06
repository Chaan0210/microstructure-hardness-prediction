"""Split a noisy-row change d (candidate - 061) into the DTA-share direction u (K-shift per unit K) and the orthogonal rest; report the gain of each
part under plain dev, non-stretch and label-matched (LM) criteria. usage: decomp_k.py SLOTSPEC... (as eval_ta4.py)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, numpy as np, nsgain as N
from blend_dev import align_dev
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; y, g, C, c61 = N.y, N.g, N.C, N.c61; w = np.load(_repo_io('lm_weights.npy'))
BASE = {'38': '../push-300/dev-{v}-v35.npz', '39': '../push-600/dev-{v}-angh.npz', '44': '../push-800/dev-{v}-anghcor.npz'}
path = lambda t, v: next(f for f in (f'../push-1200/dev-{v}-{t}.npz', f'dev-{v}-{t}.npz') if os.path.exists(f))
aw = 0.256; part = (C['ridge'] + C['logr']) / 2; other = (c61 - aw * part) / (1 - aw); u = np.where(g, aw * (part - other), 0)
def crit(nm, new):
    info, _ = N.decomp('', new, quiet=True); d = (w * ((new - y) ** 2 - (c61 - y) ** 2))[:, g].mean(0)
    rng = np.random.RandomState(0); n = g.sum(); bs = [d[rng.randint(0, n, n)].mean() for _ in range(3000)]
    print(f'  {nm:22s} plain {N.rm(c61) - N.rm(new):+.4f} non-stretch {info:+.4f} LM dMSE {d.mean():+.2f} (SE {d.std() / np.sqrt(n):.2f}, P<0 {np.mean(np.array(bs) < 0):.2f})')
for spec in sys.argv[1:]:
    delta = np.zeros((3, 400))
    for a in spec.split('+'):
        s, t = a.split(':'); delta += 0.128 * (np.mean([align_dev(path(t, v)) for v in V], 0) - np.mean([align_dev(BASE[s].format(v=v)) for v in V], 0))
    d = np.where(g, delta, 0); k = (d * u).sum() / (u * u).sum(); orth = d - k * u
    print(f'{spec}: K_eff {1 + k:.2f}, orthogonal share of change {np.sqrt((orth ** 2).sum() / (d ** 2).sum()):.2f}')
    crit('full change', c61 + d); crit('K-part only', c61 + k * u); crit('orthogonal part only', c61 + orth)
