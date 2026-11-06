"""ZM: mid-band rows (train-median noise 9.27 < noise < 12 gray, boundaries still visible) take the zero-shot v3 grain columns (same measurement as the
clean half) instead of the fine-tuned 4-view mean, and NaN for the across-view std columns (as clean rows). Noisy preset rows (>= 12) unchanged.
usage: make_zm.py SPLIT [SPLIT]  -> feat-<v>-anghcorZM-<split>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
for split in sys.argv[1:]:
    Z = np.load(_repo_path(E, f'grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); zk = [str(k) for k in Z['keys']]; zpos = {str(i): j for j, i in enumerate(Z['ids'])}
    Ztr = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); thr = float(np.nanmedian(Ztr['X'][:, [str(k) for k in Ztr['keys']].index('noise')]))
    for v in ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z'):
        F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-{split}.npz'), allow_pickle=True); K = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
        X = F['X'].astype(float).copy(); Xz = Z['X'][[zpos[i] for i in ids]].astype(float)
        n = Xz[:, zk.index('noise')]; mid = (n > thr) & (n * 255 < 12)
        for k in K:
            if k in zk: X[mid, K.index(k)] = Xz[mid, zk.index(k)]
            elif k.startswith('vstd_'): X[mid, K.index(k)] = np.nan
        np.savez(_repo_path(H, f'feat-{v}-anghcorZM-{split}.npz'), ids=F['ids'], X=X, keys=F['keys'])
        print(split, v, 'mid rows switched', int(mid.sum()))
