"""Piecewise grain model: rows with noise < THR take dev-<LOW>, others dev-<HIGH> (each may be comma list = mean). Reports pooled W0.75/W1 + terciles.
usage: split_combine.py THR LOW HIGH"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, os
def load(t):
    for d in ('.', '../push-300'):
        f = f'{d}/dev-{t}.npz'
        if os.path.exists(f): return np.load(_repo_io(f), allow_pickle=True)
thr = float(sys.argv[1]); L = [load(t) for t in sys.argv[2].split(',')]; H = [load(t) for t in sys.argv[3].split(',')]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
parts = ['original', '20261031', '20261101']; ref = list(L[0]['original_ids'])
nzr = np.nan_to_num(F['X'][[pos[i] for i in ref], keys.index('noise')], nan=99); q = np.quantile(nzr[nzr < 90], [1/3, 2/3]); t = np.digitize(nzr, q)
se = {0.75: [], 1.0: []}; pp = []
for n in parts:
    ids = list(L[0][f'{n}_ids']); y = L[0][f'{n}_y']; p = L[0][f'{n}_p021']; f = L[0][f'{n}_failed']
    nz = np.nan_to_num(F['X'][[pos[i] for i in ids], keys.index('noise')], nan=99)
    gl = np.mean([d[f'{n}_g'] for d in L], 0); gh = np.mean([d[f'{n}_g'] for d in H], 0); g = np.where(nz < thr, gl, gh); g = np.where(f, p, g)
    ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]
    for w in se: se[w].append((((1 - w) * p + w * g - y) ** 2)[j])
    pp.append(f'{np.sqrt(np.mean((0.25*p+0.75*g-y)**2)):.3f}')
e = np.mean(se[0.75], 0); e1 = np.mean(se[1.0], 0)
print(' '.join(pp), f'pooled W.75 {np.sqrt(e.mean()):.4f} W1 {np.sqrt(e1.mean()):.4f} terc W.75', ' '.join(f'{np.sqrt(e[t==k].mean()):.3f}' for k in range(3)))
