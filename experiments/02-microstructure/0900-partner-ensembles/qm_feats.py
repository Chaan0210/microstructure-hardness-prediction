"""Label-free scale alignment of the noisy rows' features to the clean rows (fixed a priori, fitted on the 500 train rows only, never on test):
reference = train rows with noise below the train median; the other train rows are split into three noise bands (terciles of their noise).
In each band every column except 'noise' is linearly mapped so that its robust centre/scale (median, IQR/1.349) equals the reference's;
test rows use the band of their noise value (train band edges) and the train-fitted maps. Rationale: on noisy rows the phase / heterogeneity
measurements are compressed (u_ph sd 0.055 vs 0.087 clean, same label distribution), so one phase->hardness relation can be shared.
usage: qm_feats.py VARIANT [SRC_SUFFIX=anghcor] [SRC_DIR=push-800] -> feat-<VARIANT>-<SRC_SUFFIX>qm-{train,test}.npz (push-900; empty suffix -> feat-<VARIANT>-qm-*)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
v = sys.argv[1]; suf = sys.argv[2] if len(sys.argv) > 2 else 'anghcor'; src = _repo_path(E, sys.argv[3] if len(sys.argv) > 3 else 'push-800')   # 038 slot: '' push-500; 039: angh push-700
stem = f'feat-{v}-{suf}' if suf else f'feat-{v}'
Tr = np.load(_repo_io(src / f'{stem}-train.npz'), allow_pickle=True); Te = np.load(_repo_io(src / f'{stem}-test.npz'), allow_pickle=True)
keys = [str(k) for k in Tr['keys']]; j = keys.index('noise'); A = Tr['X'].astype(float); T = Te['X'].astype(float)
nz = np.where(np.isnan(A[:, j]), np.nanmedian(A[:, j]), A[:, j]); med = np.median(nz); ref = nz < med
edges = np.quantile(nz[~ref], [1 / 3, 2 / 3])
def band(n): n = np.where(np.isnan(n), np.nanmedian(A[:, j]), n); return np.where(n < med, -1, np.digitize(n, edges))
def rs(x):
    x = x[np.isfinite(x)]
    if len(x) < 5: return np.nan, np.nan
    q1, q2, q3 = np.quantile(x, [.25, .5, .75]); return q2, (q3 - q1) / 1.349
def align(X, b, Xfit, bfit):
    X = X.copy()
    for c in range(X.shape[1]):
        if c == j: continue
        mr, sr = rs(Xfit[bfit == -1, c])
        for k in range(3):
            mb, sb = rs(Xfit[bfit == k, c])
            if not (np.isfinite(sb) and sb > 1e-12 and np.isfinite(sr)): continue
            m = b == k; X[m, c] = mr + (X[m, c] - mb) * sr / sb
    return X
bt = band(A[:, j]); be = band(T[:, j])
At = align(A, bt, A, bt); Tt = align(T, be, A, bt)
for nm, F, X in (('train', Tr, At), ('test', Te, Tt)):
    np.savez(_repo_path(H, f'feat-{v}-{suf}qm-{nm}.npz' if suf else f'feat-{v}-qm-{nm}.npz'), ids=F['ids'], X=X, keys=F['keys'])
print(v, 'median noise', round(med, 4), 'band edges', edges.round(4), 'test band counts', [int((be == k).sum()) for k in (-1, 0, 1, 2)])
