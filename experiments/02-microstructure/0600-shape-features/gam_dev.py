"""Spline-additive ridge model (GAM-like) on a curated column list, official dev partitions/folds. Saves dev-<tag>.npz (grain = GAM OOF).
usage: gam_dev.py FEATURE_NPZ TAG COLS(comma) [n_knots=5] [alpha grid via RidgeCV]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np
from pathlib import Path
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
P = Path(_repo_io('../..')); R = Path(_repo_io('../grain-cellpose-220'))
fpath, tag, cols = sys.argv[1], sys.argv[2], sys.argv[3].split(','); nk = int(sys.argv[4]) if len(sys.argv) > 4 else 5
F = np.load(_repo_io(fpath), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
o3 = np.load(_repo_path(R, 'oof-v3.npz'))
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
ci = [keys.index(c) for c in cols]; save = {}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; a = f'B_v3feat_v3mask_{name}_'; p = 2 * o3[a + 'b'] - o3[a + 'g']
    X = XA[[pos[i] for i in ids]]; failed = np.nan_to_num(X[:, keys.index('n_obj')], nan=0) < 20; Z = X[:, ci]
    g = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k; med = np.nanmedian(Z[tr], 0); Zi = np.where(np.isnan(Z), med, Z)
        lo, hi = np.percentile(Zi[tr], 1, 0), np.percentile(Zi[tr], 99, 0); Zi = np.clip(Zi, lo, hi)
        m = make_pipeline(StandardScaler(), SplineTransformer(n_knots=nk, degree=3), RidgeCV(alphas=np.logspace(-2, 4, 30))); m.fit(Zi[tr], y[tr]); g[va] = m.predict(Zi[va])
    print(name, 'gam', round(float(np.sqrt(np.mean((g - y) ** 2))), 3), flush=True)
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_fold': fold, f'{name}_p021': p, f'{name}_g': g, f'{name}_failed': failed})
np.savez(_repo_io(f'dev-{tag}.npz'), **save)
