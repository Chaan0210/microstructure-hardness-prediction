"""v2 grain model: fixed transform + family selection by inner CV (plan-v2.json)."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingRegressor

W = 0.5          # fixed blend weight, identical in validation and final
MIN_OBJ = 20     # segmentation-failure fallback


def logclip(A):
    return np.log1p(np.clip(A, 0, None))  # all columns, fixed by definition


def ridge():
    return make_pipeline(SimpleImputer(strategy='median'), FunctionTransformer(logclip), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def krr():
    return GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), FunctionTransformer(logclip), StandardScaler(), KernelRidge(kernel='rbf')),
                        {'kernelridge__alpha': [0.03, 0.1, 0.3, 1.0], 'kernelridge__gamma': [0.005, 0.01, 0.02, 0.05]},
                        cv=KFold(4, shuffle=True, random_state=0), scoring='neg_mean_squared_error')


def hgb():
    return make_pipeline(SimpleImputer(strategy='median'),
                         HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=15, l2_regularization=1.0, random_state=0))


FAMILIES = {'ridge': ridge, 'krr': krr, 'hgb': hgb}


def fit_predict(make, Xa, ya, Xb):
    mu = ya.mean(); return make().fit(Xa, ya - mu).predict(Xb) + mu


def select_and_fit(X, y, inner_folds):
    """Pick a family by inner-CV standalone RMSE on (X, y) only, refit it on all of (X, y)."""
    scores = {}
    for name, make in FAMILIES.items():
        o = np.zeros(len(y))
        for j in np.unique(inner_folds):
            a, v = inner_folds != j, inner_folds == j
            o[v] = fit_predict(make, X[a], y[a], X[v])
        scores[name] = float(np.sqrt(np.mean((o - y) ** 2)))
    best = min(scores, key=scores.get)
    mu = y.mean(); model = FAMILIES[best]().fit(X, y - mu)
    return best, scores, (lambda Xn: model.predict(Xn) + mu), model
