"""Scan stored OOF files (original partition, 400 dev ids): RMSE, error correlation with 039 grain OOF, and gain when added as a third blend component
with fold-wise cross-validated (nested over the 5 outer folds) nonnegative weights."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, glob, itertools
D = np.load(_repo_io('push-600/dev-u1dmavgH2Z-angh.npz'), allow_pickle=True)
D4 = [np.load(_repo_io(f'push-600/dev-{v}-angh.npz'), allow_pickle=True) for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']]
ids = [str(i) for i in D['original_ids']]; y = D['original_y']; fold = D['original_fold']; p021 = D['original_p021']; failed = D['original_failed']
g = np.mean([d['original_g'] for d in D4], 0); g = np.where(failed, p021, g)
base = 0.25 * p021 + 0.75 * g; rm = lambda v: np.sqrt(np.mean((v - y) ** 2))
print('039 grain %.3f  021 %.3f  blend %.3f' % (rm(g), rm(p021), rm(base)))
pos = {i: k for k, i in enumerate(ids)}; rows = []
for f in sorted(glob.glob('*/**/*OOF*.npz', recursive=True) + glob.glob('*/*oof*.npz')):
    if '.venv' in f or 'push-' in f: continue
    try: d = np.load(_repo_io(f), allow_pickle=True)
    except Exception: continue
    if 'ids' not in d.files: continue
    fid = [str(i) for i in d['ids']]
    if len(set(fid) & set(ids)) < 400: continue
    j = [fid.index(i) for i in ids]
    for k in d.files:
        a = d[k]
        if not (isinstance(a, np.ndarray) and a.dtype.kind == 'f' and a.ndim == 1 and len(a) == len(fid)): continue
        p = a[j]
        if k in ('y',) or np.std(p) < 3 or rm(p) > 20 or rm(p) < 5: continue
        rows.append((f + ':' + k, p))
print('candidates', len(rows))
res = []
for name, p in rows:
    e_c = np.corrcoef(p - y, g - y)[0, 1]; e_021 = np.corrcoef(p - y, p021 - y)[0, 1]
    # nested blend: for each outer fold, choose weights (grid) on the other 4 folds, apply to held-out fold; weights w021, wp with wg = 1 - w021 - wp
    out = np.zeros_like(y); grid = [(a, b) for a in np.arange(0, 0.55, 0.05) for b in np.arange(0, 0.55, 0.05) if a + b <= 0.6]
    for k in range(5):
        tr = fold != k; best = min(grid, key=lambda w: np.mean(((1 - w[0] - w[1]) * g[tr] + w[0] * p021[tr] + w[1] * p[tr] - y[tr]) ** 2))
        out[fold == k] = (1 - best[0] - best[1]) * g[fold == k] + best[0] * p021[fold == k] + best[1] * p[fold == k]
    res.append((rm(out) - rm(base), name, rm(p), e_c, e_021))
res.sort()
for r in res[:25]: print('nested3 %.4f vs base %.4f (d %+.3f) | %-70s rmse %.2f errcorr(039) %.2f errcorr(021) %.2f' % (rm(base) + r[0], rm(base), r[0], r[1][-70:], r[2], r[3], r[4]))
