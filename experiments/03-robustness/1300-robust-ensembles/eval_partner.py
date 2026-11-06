"""Screen a new partner (npz with <key>_0 plain OOF and <key>_1 DTA OOF, [3, 400]) on top of 061 (train-only dev).
partner = DTA version on noise >= 12 rows (as 061's Ridge partners), plain elsewhere; new blend = (1-a)*061 + a*partner (failed rows unchanged).
Reports standalone RMSE by band, error correlation with 061, per-band dMSE (SE), partitions, noisy non-stretch/LM (robust_eval), and a cross-fitted share
(a chosen on 4 folds of each partition, applied to the 5th). usage: eval_partner.py file.npz key"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
import nsgain as N
from robust_eval import report
y, g, c61, nz, fail, fold = N.y, N.g, N.c61, N.nz, N.fail, N.fold
Z = np.load(_repo_io(sys.argv[1])); key = sys.argv[2]
p = np.where(g, Z[f'{key}_1'], Z[f'{key}_0']); p = np.where(fail, c61, p)
f = fail[0]; bands = {'clean<11': (nz < 11) & ~f, '11-12': (nz >= 11) & (nz < 12) & ~f, 'noisy>=12': g}
rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
print(f'{sys.argv[1]} {key}: alone ' + ' '.join(f'{k} {rb(p, m):.3f}' for k, m in bands.items()) + f' | 061 ' + ' '.join(f'{rb(c61, m):.3f}' for m in bands.values()))
for nm, comp in N.C.items(): print(f'   err corr with {nm:5s} {np.corrcoef((p - y)[:, ~f].ravel(), (comp - y)[:, ~f].ravel())[0, 1]:.3f}', end='')
print(f'\n   err corr with 061 {np.corrcoef((p - y)[:, ~f].ravel(), (c61 - y)[:, ~f].ravel())[0, 1]:.3f}')
for a in (0.05, 0.1, 0.15, 0.2, 0.3):
    new = c61 + a * (p - c61); d = (new - y) ** 2 - (c61 - y) ** 2
    cells = ' '.join(f'{k} {d[:, m].mean():+.2f}({d[:, m].mean(0).std() / np.sqrt(m.sum()):.2f})' for k, m in bands.items())
    report(f'a {a:.2f} | {cells} |', new)
# cross-fitted share: per partition, choose a on 4 folds (grid), apply to the held-out fold
grid = np.linspace(0, 0.6, 61); cf = c61.copy(); ch = []
for k in range(3):
    for fo in range(5):
        tr = (fold[k] != fo) & ~f; te = fold[k] == fo
        d = p[k] - c61[k]; r = c61[k] - y; a = grid[np.argmin([((r[tr] + aa * d[tr]) ** 2).mean() for aa in grid])]; ch.append(a); cf[k, te] = c61[k, te] + a * d[te]
report(f'cross-fitted a (mean {np.mean(ch):.3f}, sd {np.std(ch):.3f})', np.where(fail, c61, cf))
