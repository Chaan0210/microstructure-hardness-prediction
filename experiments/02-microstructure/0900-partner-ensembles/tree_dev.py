"""Tree partners on top of 054 (dev, Study900 protocol): ExtraTrees / RandomForest / LightGBM on the 98-column average-variant matrix (as diverse_dev),
fixed a-priori settings (no per-fold tuning). Reports standalone RMSE, error correlation with c054, and (1-a)*c054 + a*model for a in 0.05..0.2.
usage: tree_dev.py NAME[,NAME...]   -> tree-<NAME>.npz (OOF [3, 400])"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from blend_dev import load, rmse, parts
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
B = load(); ids, y, FOLD, fail = B['ids'], B['y'], B['fold'], B['fail']
R = np.load(_repo_path(E, 'push-800/eda/resid045.npz'), allow_pickle=True); rpos = {str(i): k for k, i in enumerate(R['ids'])}; X = R['X'][[rpos[i] for i in ids]].astype(float)
MODELS = {
    'et': lambda s: make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(1000, max_features=0.33, min_samples_leaf=2, n_jobs=4, random_state=s)),
    'rf': lambda s: make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(1000, max_features=0.33, min_samples_leaf=3, n_jobs=4, random_state=s)),
    'lgb': lambda s: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=8, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                       reg_lambda=1.0, random_state=s, verbose=-1, n_jobs=4),
}
for name in sys.argv[1].split(','):
    out = np.zeros_like(y[None].repeat(3, 0))
    for k in range(3):
        for f in range(5):
            tr, va = FOLD[k] != f, FOLD[k] == f
            out[k, va] = MODELS[name](100 * k + f).fit(X[tr], y[tr]).predict(X[va])
    np.savez(_repo_path(H, f'tree-{name}.npz'), ids=np.array(ids), oof=out)
    m = np.where(fail, B['p021'], out); c = B['c054']
    corr = np.corrcoef((m - y).ravel(), (c - y).ravel())[0, 1]; d = m - c; aopt = -np.mean(d * (c - y)) / np.mean(d * d)
    line = [f'{name}: alone {rmse(m, y):.3f} corr {corr:.3f} opt a {aopt:.3f}']
    for a in (0.05, 0.1, 0.15, 0.2):
        b = c + a * d; line.append(f'a{a}: {rmse(b, y):.4f} ({rmse(c, y) - rmse(b, y):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in parts(c, y) - parts(b, y)) + ')')
    print('\n  '.join(line), flush=True)
