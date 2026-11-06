"""Evaluate EfficientNetV2 OOF (cnnv2-*.npz, mean over given files) against CNN021 inside the 073 dev blend, on the partitions/rows available.
(a) replace CNN021 (weight 0.16); (b) extra partner new = (1-a)*073 + a*V2, a cross-fitted per partition (chosen on 4 folds, applied to the 5th)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, nsgain as N
y, C, nz, fail, fold, W = N.y, N.C, N.nz, N.fail, N.fold, N.W; f = fail[0]
V = np.nanmean([np.load(_repo_io(fn))['oof'] for fn in sys.argv[1:]], 0); c73 = np.load(_repo_io('dev-nb-candidate.npy'))
avail = np.isfinite(V).all(1); parts = np.where(avail)[0]; print('files', sys.argv[1:], 'complete partitions', parts.tolist())
bands = {'clean<11': (nz < 11) & ~f, '11-12': (nz >= 11) & (nz < 12) & ~f, 'noisy>=12': (nz >= 12) & ~f, 'all': ~f}
rb = lambda a, m, ps: np.sqrt(((a[ps] - y) ** 2)[:, m].mean())
for nm, m in bands.items(): print(f'  {nm:9s} V2 {rb(V, m, parts):.3f} | CNN021 {rb(C["cnn"], m, parts):.3f} | 073 {rb(c73, m, parts):.3f}')
r73 = (c73 - y)[parts][:, ~f].ravel(); print('  error corr with 073: V2 %.3f, CNN021 %.3f; V2 vs CNN021 %.3f' % (np.corrcoef((V - y)[parts][:, ~f].ravel(), r73)[0, 1], np.corrcoef((C['cnn'] - y)[parts][:, ~f].ravel(), r73)[0, 1], np.corrcoef((V - y)[parts][:, ~f].ravel(), (C['cnn'] - y)[parts][:, ~f].ravel())[0, 1]))
rep = np.where(fail, c73, c73 + 0.16 * (V - C['cnn']))
def show(nm, new):
    d = (new - y) ** 2 - (c73 - y) ** 2; pp = [np.sqrt(((c73[p] - y) ** 2).mean()) - np.sqrt(((new[p] - y) ** 2).mean()) for p in parts]
    print(f'  {nm:28s} gain {np.sqrt(((c73[parts] - y) ** 2).mean()) - np.sqrt(((new[parts] - y) ** 2).mean()):+.4f} parts {np.round(pp, 3)} | ' + ' '.join(f'{k} {d[parts][:, m].mean():+.2f}' for k, m in bands.items() if k != 'all'))
show('replace CNN021 (0.16)', rep)
for a in (0.05, 0.1, 0.15, 0.2, 0.3): show(f'add V2 a={a}', np.where(fail, c73, c73 + a * (V - c73)))
grid = np.linspace(0, 0.6, 61); cf = c73.copy(); ch = []
for p in parts:
    for k in range(5):
        tr = (fold[p] != k) & ~f; te = fold[p] == k; d = V[p] - c73[p]; r = c73[p] - y
        a = grid[np.argmin([((r[tr] + aa * d[tr]) ** 2).mean() for aa in grid])]; ch.append(a); cf[p, te] = np.where(f[te], c73[p, te], c73[p, te] + a * d[te])
show(f'cross-fitted add (a mean {np.mean(ch):.2f})', cf)
