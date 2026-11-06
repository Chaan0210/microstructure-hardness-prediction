"""065 dev blend with the partner pair (RidgeCV, logRidge; gate noise>=12) swapped from sr-A-d1to7n12-wal to sr-<TAG>.npz. usage: eval_partner065.py TAG [TAG ...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
D = np.load(_repo_io('dev065.npz')); y, nz, fail, c = D['y'], D['nz'], D['fail'], D['c']; g = (nz >= 12) & ~fail; S0 = np.load(_repo_io('sr-A-d1to7n12-wal.npz'))
r = lambda a, m=slice(None): np.sqrt(((a - y)[:, m] ** 2).mean())
print(f'065 {r(c):.4f} parts {np.round(np.sqrt(((c - y) ** 2).mean(1)), 4)} noisy {r(c, g):.3f}')
for t in sys.argv[1:]:
    S = np.load(_repo_io(f'sr-{t}.npz')); cn = c + 0.128 * g * ((S['ridge'] - S0['ridge']) + (S['logridge'] - S0['logridge']))
    print(f'{t:34s} {r(cn):.4f} parts {np.round(np.sqrt(((cn - y) ** 2).mean(1)), 4)} noisy {r(cn, g):.3f} | ridge noisy {r(S["ridge"], g):.3f} logridge {r(S["logridge"], g):.3f} (062: {r(S0["ridge"], g):.3f} {r(S0["logridge"], g):.3f})')
