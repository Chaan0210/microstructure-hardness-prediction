"""usage: combine.py TAG[,TAG...] [TAG,...]  -- each arg is a candidate = mean of listed dev OOFs (looked up in push-600 then push-300 as dev-<tag>.npz)
prints per-partition grain/W0.75/W1 and pooled values plus per-noise-tercile pooled W0.75."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, os
def load(t):
    for d in ('.', '../push-300'):
        f = f'{d}/dev-{t}.npz'
        if os.path.exists(f): return np.load(_repo_io(f), allow_pickle=True)
    raise FileNotFoundError(t)
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
parts = ['original', '20261031', '20261101']
for arg in sys.argv[1:]:
    D = [load(t) for t in arg.split(',')]; ref = list(D[0]['original_ids'])
    noise = np.nan_to_num(F['X'][[pos[i] for i in ref], keys.index('noise')], nan=99); q = np.quantile(noise[noise < 90], [1/3, 2/3]); t = np.digitize(noise, q)
    se = {w: [] for w in (0.75, 1.0)}; out = []
    for n in parts:
        ids = list(D[0][f'{n}_ids']); y = D[0][f'{n}_y']; p = D[0][f'{n}_p021']; f = D[0][f'{n}_failed']
        g = np.mean([d[f'{n}_g'] for d in D], 0); g = np.where(f, p, g); ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]
        for w in se: se[w].append((((1 - w) * p + w * g - y) ** 2)[j])
        out.append(f'{np.sqrt(np.mean((g-y)**2)):.3f}/{np.sqrt(np.mean((0.25*p+0.75*g-y)**2)):.3f}')
    e75 = np.mean(se[0.75], 0); e1 = np.mean(se[1.0], 0)
    print(f'{arg:60s} parts(grain/W.75) {" ".join(out)} | pooled W.75 {np.sqrt(e75.mean()):.4f} W1 {np.sqrt(e1.mean()):.4f} | terc W.75 ' + ' '.join(f'{np.sqrt(e75[t==k].mean()):.3f}' for k in range(3)))
