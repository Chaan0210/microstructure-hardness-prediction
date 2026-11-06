"""Q1: same 97 columns as 039 (avg variant), official dev partitions/folds. Classic ML regressors vs TabPFN-3.5 (existing 3-seed OOF).
Hyperparameters fixed a priori for small data; boosting rounds by early stopping on a 20% split INSIDE the outer-training rows, then refit.
Saves OOF per model and partition to mlcmp-oof.npz."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, warnings, time
warnings.filterwarnings('ignore')
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold
import lightgbm as lgb, xgboost as xgb, catboost as cb
D = np.load(_repo_io('../push-600/dev-u1dmavgH2Z-angh.npz'), allow_pickle=True)
F = np.load(_repo_io('feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); fp = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
parts = ['original', '20261031', '20261101']
def es_split(n, seed): r = np.random.RandomState(seed); p = r.permutation(n); k = int(0.8 * n); return p[:k], p[k:]
def m_ridge(Xtr, ytr, Xva, s):
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 40))).fit(Xtr, ytr); return m.predict(Xva)
def m_svr(Xtr, ytr, Xva, s):
    base = make_pipeline(SimpleImputer(strategy='median'), QuantileTransformer(n_quantiles=150, output_distribution='normal'), StandardScaler(), SVR())
    g = GridSearchCV(base, {'svr__C': [10, 30, 100], 'svr__gamma': [0.003, 0.01], 'svr__epsilon': [1.0]}, cv=KFold(4, shuffle=True, random_state=s), scoring='neg_mean_squared_error').fit(Xtr, ytr)
    return g.predict(Xva)
def m_rf(Xtr, ytr, Xva, s):
    m = make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(500, min_samples_leaf=3, max_features=0.33, n_jobs=8, random_state=s)).fit(Xtr, ytr); return m.predict(Xva)
def m_et(Xtr, ytr, Xva, s):
    m = make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(800, min_samples_leaf=2, max_features=0.5, n_jobs=8, random_state=s)).fit(Xtr, ytr); return m.predict(Xva)
def m_lgb(Xtr, ytr, Xva, s):
    a, b = es_split(len(ytr), s); P = dict(objective='regression', learning_rate=0.02, num_leaves=7, min_child_samples=12, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=s, n_jobs=8)
    m = lgb.train(P, lgb.Dataset(Xtr[a], ytr[a]), 3000, valid_sets=[lgb.Dataset(Xtr[b], ytr[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
    n = max(50, int(m.best_iteration * 1.1)); return lgb.train(P, lgb.Dataset(Xtr, ytr), n).predict(Xva)
def m_xgb(Xtr, ytr, Xva, s):
    a, b = es_split(len(ytr), s); P = dict(n_estimators=3000, learning_rate=0.02, max_depth=3, min_child_weight=3, subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0, random_state=s, n_jobs=8, early_stopping_rounds=150)
    m = xgb.XGBRegressor(**P).fit(Xtr[a], ytr[a], eval_set=[(Xtr[b], ytr[b])], verbose=False); n = max(50, int(m.best_iteration * 1.1))
    P.pop('early_stopping_rounds'); P['n_estimators'] = n; return xgb.XGBRegressor(**P).fit(Xtr, ytr).predict(Xva)
def m_cb(Xtr, ytr, Xva, s):
    a, b = es_split(len(ytr), s); P = dict(iterations=4000, learning_rate=0.03, depth=4, l2_leaf_reg=5, loss_function='RMSE', random_seed=s, verbose=False, thread_count=8)
    m = cb.CatBoostRegressor(**P, od_type='Iter', od_wait=200).fit(Xtr[a], ytr[a], eval_set=(Xtr[b], ytr[b]), use_best_model=True); n = max(50, int(m.get_best_iteration() * 1.1))
    P['iterations'] = n; return cb.CatBoostRegressor(**P).fit(Xtr, ytr).predict(Xva)
models = dict(ridge=m_ridge, svr=m_svr, rf=m_rf, et=m_et, lgbm=m_lgb, xgb=m_xgb, catboost=m_cb)
OOF = {}
for p in parts:
    ids = [str(i) for i in D[p + '_ids']]; y = D[p + '_y']; fold = D[p + '_fold']; X = XA[[fp[i] for i in ids]]
    for name, fn in models.items():
        t0 = time.time(); o = np.zeros(len(y))
        for k in range(5):
            tr, va = fold != k, fold == k
            o[va] = np.mean([fn(X[tr], y[tr], X[va], 10 * k + sd) for sd in range(2 if name in ('lgbm', 'xgb', 'catboost', 'rf', 'et') else 1)], 0)
        OOF[f'{p}|{name}'] = o; print(p, name, 'grain rmse %.3f' % np.sqrt(np.mean((o - y) ** 2)), '%.0fs' % (time.time() - t0), flush=True)
np.savez(_repo_io('mlcmp-oof.npz'), **OOF)
