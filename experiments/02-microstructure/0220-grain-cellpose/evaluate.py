"""Honest evaluation of Cellpose grain features on dev400 (original 5 folds, same IDs as best021 OOF).

All supervised fits (imputation, scaling, alpha/kernel selection, blend weights) happen inside training folds only.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, sys, warnings
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')

R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(R))
from features import feature_matrix

feat_path = _repo_path(R, 'features-train.npz')
if not feat_path.exists():
    z = np.load(_repo_path(R, 'masks-train.npz'))
    ids_all, masks = z['ids'], z['masks']
    imgs = [np.array(Image.open(_repo_path(P, f'data/train/{i}.png'))) for i in ids_all]
    X_all, keys = feature_matrix(imgs, masks)
    np.savez(_repo_io(feat_path), ids=ids_all, X=X_all, keys=np.array(keys))
F = np.load(_repo_io(feat_path)); keys = list(F['keys']); pos = {i: k for k, i in enumerate(F['ids'])}

meta = np.load(_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'))
ids, y, fold = meta['ids'], meta['y'], meta['fold']
p021, cnn256 = meta['fixed_control'], meta['native']
X = F['X'][[pos[i] for i in ids]]
r021 = y - p021
rm = lambda a: float(np.sqrt(np.mean((a - y) ** 2)))
report = {'n_dev': int(len(y)), 'n_features': len(keys), 'nan_rows': int(np.isnan(X).any(1).sum()), 'P021_oof': rm(p021)}

# ---- descriptive correlations (label used only for reporting, not for any selection) ----
corr = {}
for j, k in enumerate(keys):
    ok = ~np.isnan(X[:, j])
    corr[k] = dict(rho_y=round(float(spearmanr(X[ok, j], y[ok])[0]), 3), rho_resid021=round(float(spearmanr(X[ok, j], r021[ok])[0]), 3))
report['spearman'] = corr

POSITIVE = [j for j, k in enumerate(keys) if np.nanmin(F['X'][:, j]) >= 0]
def logpos(A):
    A = A.copy(); A[:, POSITIVE] = np.log1p(A[:, POSITIVE]); return A
LT = FunctionTransformer(logpos)

def ridge(): return make_pipeline(SimpleImputer(strategy='median'), LT, StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
def krr(): return GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), LT, StandardScaler(), KernelRidge(kernel='rbf')),
                               {'kernelridge__alpha': [0.03, 0.1, 0.3, 1.0], 'kernelridge__gamma': [0.005, 0.01, 0.02, 0.05]},
                               cv=KFold(4, shuffle=True, random_state=0), scoring='neg_mean_squared_error')
def hgb(): return make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=15, l2_regularization=1.0, random_state=0))

def oof(make, Xm, target):
    o = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k
        mu = target[tr].mean(); m = make().fit(Xm[tr], target[tr] - mu); o[va] = m.predict(Xm[va]) + mu
    return o

def honest_blend(base, cand):
    out = np.zeros(len(y)); ws = []
    for k in range(5):
        tr, va = fold != k, fold == k; g = np.linspace(0, 1, 101)
        w = g[np.argmin([np.mean(((1 - a) * base[tr] + a * cand[tr] - y[tr]) ** 2) for a in g])]
        ws.append(float(w)); out[va] = (1 - w) * base[va] + w * cand[va]
    return out, ws

res = {}
preds = {}
for name, make in [('ridge', ridge), ('krr', krr), ('hgb', hgb)]:
    preds[name] = oof(make, X, y); res[f'grain_{name}_standalone'] = rm(preds[name])
preds['grain_mean3'] = (preds['ridge'] + preds['krr'] + preds['hgb']) / 3; res['grain_mean3_standalone'] = rm(preds['grain_mean3'])
for name in ['ridge', 'krr', 'hgb', 'grain_mean3']:
    b, ws = honest_blend(p021, preds[name]); res[f'blend_P021+{name}'] = rm(b); res[f'blend_P021+{name}_weights'] = ws
    preds[f'blend_{name}'] = b
# stacking: P021 as an extra input to ridge/krr
XS = np.column_stack([X, p021])
for name, make in [('ridge', ridge), ('krr', krr)]:
    preds[f'stack_{name}'] = oof(make, XS, y); res[f'stack_P021+feats_{name}'] = rm(preds[f'stack_{name}'])
# residual correction of P021
for name, make in [('ridge', ridge), ('krr', krr)]:
    rc = oof(make, X, r021); preds[f'resid_{name}'] = p021 + rc; res[f'resid_{name}'] = rm(preds[f'resid_{name}'])
res['err_corr_grain_mean3_vs_P021'] = float(np.corrcoef(y - preds['grain_mean3'], r021)[0, 1])
res['err_corr_grain_mean3_vs_CNN256'] = float(np.corrcoef(y - preds['grain_mean3'], y - cnn256)[0, 1])
report['results'] = res

# ---- paired bootstrap for the best non-standalone variant ----
cands = {k: v for k, v in preds.items() if k.startswith(('blend_', 'stack_', 'resid_'))}
best = min(cands, key=lambda k: rm(cands[k]))
rng = np.random.default_rng(20261007); gains = []
for _ in range(10000):
    i = rng.integers(0, len(y), len(y))
    gains.append(np.sqrt(np.mean((p021[i] - y[i]) ** 2)) - np.sqrt(np.mean((cands[best][i] - y[i]) ** 2)))
report['best_variant'] = dict(name=best, rmse=rm(cands[best]), gain=rm(p021) - rm(cands[best]),
                              ci95=[float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))],
                              folds_improved=int(sum(rm_k < rb_k for rm_k, rb_k in [(np.sqrt(np.mean((cands[best][fold == k] - y[fold == k]) ** 2)), np.sqrt(np.mean((p021[fold == k] - y[fold == k]) ** 2))) for k in range(5)])))

# ---- error concentration re-check with true (segmented) grain size and quality ----
conc = {}
for k in ['grain_eqd_mean', 'coverage', 'noise', 'f_dark_35', 'ar_wmean']:
    x = X[:, keys.index(k)]; qs = np.nanquantile(x, [1 / 3, 2 / 3]); g = np.digitize(x, qs)
    conc[k] = [dict(tercile=t, rmse_P021=round(float(np.sqrt(np.mean(r021[g == t] ** 2))), 2), y_std=round(float(y[g == t].std()), 2),
                    rmse_best=round(float(np.sqrt(np.mean((y - cands[best])[g == t] ** 2))), 2)) for t in range(3)]
report['error_by_tercile'] = conc
np.savez(_repo_path(R, 'oof-predictions.npz'), ids=ids, y=y, fold=fold, p021=p021, **{k: v for k, v in preds.items()})
json.dump(report, open(_repo_path(R, 'results.json'), 'w'), indent=1)
print(json.dumps({k: report[k] for k in ['n_dev', 'n_features', 'nan_rows', 'P021_oof', 'results', 'best_variant']}, indent=1))
