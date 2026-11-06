"""usage: blend.py BASE_TAGS(comma) EXTRA_TAGS(comma) w1,w2,...  -> grain = (1-w)*mean(base)+w*mean(extra); reports pooled W0.75 blend with best021"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, os
def load(t):
    for d in ('.', '../push-300'):
        f = f'{d}/dev-{t}.npz'
        if os.path.exists(f): return np.load(_repo_io(f), allow_pickle=True)
B = [load(t) for t in sys.argv[1].split(',')]; E = [load(t) for t in sys.argv[2].split(',')]; ws = [float(w) for w in sys.argv[3].split(',')]
parts = ['original', '20261031', '20261101']; ref = list(B[0]['original_ids'])
for w in ws:
    se = []; pp = []
    for n in parts:
        ids = list(B[0][f'{n}_ids']); y = B[0][f'{n}_y']; p = B[0][f'{n}_p021']; f = B[0][f'{n}_failed']
        gb = np.mean([d[f'{n}_g'] for d in B], 0); ge = np.mean([d[f'{n}_g'] for d in E], 0); g = np.where(f, p, (1 - w) * gb + w * ge)
        b = 0.25 * p + 0.75 * g; ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]; se.append(((b - y) ** 2)[j]); pp.append(f'{np.sqrt(np.mean((b-y)**2)):.3f}')
    print(w, ' '.join(pp), 'pooled', round(float(np.sqrt(np.mean(se, 0).mean())), 4))
