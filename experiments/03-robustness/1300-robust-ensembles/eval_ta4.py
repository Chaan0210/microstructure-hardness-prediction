"""061 + noisy-row replacement of the TabPFN slots by TA variants (dev files dev-<v>-<tag>.npz), same weight (0.128 per slot).
usage: eval_ta4.py 44:anghcorTA4 39:anghTA4 38:v35TA4   (missing variant files -> slot skipped); also prints single-variant pilots."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, numpy as np, nsgain as N
from blend_dev import align_dev
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; y, g, c61 = N.y, N.g, N.c61; w = np.load(_repo_io('lm_weights.npy'))
BASE = {'38': '../push-300/dev-{v}-v35.npz', '39': '../push-600/dev-{v}-angh.npz', '44': '../push-800/dev-{v}-anghcor.npz'}
def path(t, v):
    for d in ('../push-1200', '.'):
        f = f'{d}/dev-{v}-{t}.npz'
        if os.path.exists(f): return f
rn = lambda a: float(np.sqrt(((a - y)[:, g] ** 2).mean()))
delta = np.zeros((3, 400)); used = []
for a in sys.argv[1:]:
    s, t = a.split(':'); fs = [path(t, v) for v in V]
    for v, f in zip(V, fs):
        if f: print(f'  slot {s} {v}: base noisy {rn(align_dev(BASE[s].format(v=v))):.3f} -> {t} {rn(align_dev(f)):.3f}')
    if all(fs):
        new = np.mean([align_dev(f) for f in fs], 0); old = np.mean([align_dev(BASE[s].format(v=v)) for v in V], 0); delta += 0.128 * (new - old); used.append(a)
if used:
    c = np.where(g, c61 + delta, c61); N.decomp('061 + ' + ' + '.join(used), c)
    dm = (w * ((c - y) ** 2 - (c61 - y) ** 2))[:, g].mean(0); print('   LM noisy dMSE %+.2f (SE %.2f)' % (dm.mean(), dm.std() / np.sqrt(g.sum())))
