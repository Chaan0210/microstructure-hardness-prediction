"""v3 grain models (plan-v3.json). Same structure as models_v2 with a wider KRR grid and a signed-column exception."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingRegressor

W = 0.5
MIN_OBJ = 20
SIGNED = ['logA_skew']


def make_transform(keys):
    signed = np.array([k in SIGNED for k in keys])
    def tf(A):
        A = A.copy(); A[:, ~signed] = np.log1p(np.clip(A[:, ~signed], 0, None)); return A
    return FunctionTransformer(tf)


def families(keys):
    T = lambda: make_transform(keys)
    def ridge():
        return make_pipeline(SimpleImputer(strategy='median'), T(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    def krr():
        return GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), T(), StandardScaler(), KernelRidge(kernel='rbf')),
                            {'kernelridge__alpha': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0], 'kernelridge__gamma': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]},
                            cv=KFold(4, shuffle=True, random_state=0), scoring='neg_mean_squared_error')
    def hgb():
        return make_pipeline(SimpleImputer(strategy='median'),
                             HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=15, l2_regularization=1.0, random_state=0))
    return {'ridge': ridge, 'krr': krr, 'hgb': hgb}


def fit_predict(make, Xa, ya, Xb):
    mu = ya.mean(); return make().fit(Xa, ya - mu).predict(Xb) + mu


def select_and_fit(X, y, inner_folds, keys):
    fam = families(keys); scores = {}
    for name, make in fam.items():
        o = np.zeros(len(y))
        for j in np.unique(inner_folds):
            a, v = inner_folds != j, inner_folds == j
            o[v] = fit_predict(make, X[a], y[a], X[v])
        scores[name] = float(np.sqrt(np.mean((o - y) ** 2)))
    best = min(scores, key=scores.get)
    mu = y.mean(); model = fam[best]().fit(X, y - mu)
    return best, scores, (lambda Xn: model.predict(Xn) + mu), model
