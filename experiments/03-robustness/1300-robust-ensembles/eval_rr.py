"""Blend effect of rebuilt linear partners (rr-<mode>.npz) on 061; 'none' must reproduce 061 (check)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, nsgain as N
from robust_eval import report
y, g, c61, C, nz, fail = N.y, N.g, N.c61, N.C, N.nz, N.fail; f = fail[0]; cl = (nz < 11) & ~f; mid = (nz >= 11) & (nz < 12) & ~f
rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
for mode in sys.argv[1:]:
    Z = np.load(_repo_io(f'rr-{mode}.npz')); r = np.where(g, Z['ridge_1'], Z['ridge_0']); l = np.where(g, Z['logridge_1'], Z['logridge_0'])
    Cx = dict(C); Cx['ridge'] = r; Cx['logr'] = l; new = N.blend(Cx); d = (new - y) ** 2 - (c61 - y) ** 2
    print(f'{mode:5s} ridge clean {rb(r, cl):.3f} mid {rb(r, mid):.3f} noisy {rb(r, g):.3f} | logr clean {rb(l, cl):.3f} mid {rb(l, mid):.3f} noisy {rb(l, g):.3f} | '
          f'clean dMSE {d[:, cl].mean():+.2f} ({d[:, cl].mean(0).std() / np.sqrt(cl.sum()):.2f}) mid {d[:, mid].mean():+.2f} noisy {d[:, g].mean():+.2f} ({d[:, g].mean(0).std() / np.sqrt(g.sum()):.2f}) max|d 061| {np.abs(new - c61).max():.2e}')
    report(f'   {mode}', new)
