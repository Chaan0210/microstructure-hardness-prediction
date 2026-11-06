"""Honest per-noise-segment shrinkage on dev: c = g + lam_s * (c054 - g) (lam=1 is 054) and c = c054 + mu_s * (021 - c054), lam_s/mu_s fitted
on the other four folds of the same partition (1 parameter per segment). Segments from the production 'noise' column (median / terciles)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from blend_dev import *
B = load(9); y, F, fail = B['y'], B['fold'], B['fail']; c = B['c054']; g = np.where(fail, B['p021'], B['g']); p = B['p021']
NZ = np.load(_repo_io('/private/tmp/claude-501/-Users-dongchan-Desktop-PythonWorkspace-microstructure-2026/f62e347b-cb1a-4c17-8d54-2bf505be6706/scratchpad/nz_dev.npy')); NZ = np.where(np.isnan(NZ), np.nanmedian(NZ), NZ)
def honest(base, dirn, seg):
    P = base.copy(); L = []
    for k in range(3):
        for f in range(5):
            tr, va = (F[k] != f) & ~fail[k], F[k] == f
            for s in np.unique(seg):
                mt, mv = tr & (seg == s), va & (seg == s); d = dirn[k] - base[k]
                lam = -np.sum(d[mt] * (base[k][mt] - y[mt])) / np.sum(d[mt] ** 2); L.append((s, lam)); P[k, mv] = base[k][mv] + lam * d[mv]
    return P, L
segs = {'one': np.zeros(400, int), 'median': (NZ >= np.median(NZ)).astype(int), 'tercile': np.digitize(NZ, np.quantile(NZ, [1 / 3, 2 / 3]))}
print('054', round(rmse(c, y), 4))
for sn, seg in segs.items():
    for nm, base, dirn in (('toward g', c, g), ('toward 021', c, p)):
        P, L = honest(base, dirn, seg); L = np.array(L)
        lm = ' '.join(f's{int(s)}:{L[L[:, 0] == s, 1].mean():+.2f}' for s in np.unique(L[:, 0]))
        print(f'{sn:8s} {nm:10s}: {rmse(P, y):.4f} ({rmse(c, y) - rmse(P, y):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in parts(c, y) - parts(P, y)) + f')  coef {lm}')
