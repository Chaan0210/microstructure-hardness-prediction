"""Re-run the grain-KRR + best021 blend on the original split and the two existing re-splits (20261031, 20261101).

Same fixed recipe everywhere: KRR(RBF) on 36 grain features with inner 4-fold grid, fold-honest blend weight with that
split's best021 OOF, label-free fallback to best021 when segmentation failed (n_obj < 20).
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, sys, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(R))
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold

F = np.load(_repo_path(R, 'features-train.npz')); keys = list(F['keys']); pos = {i: k for k, i in enumerate(F['ids'])}
POSITIVE = [j for j in range(len(keys)) if np.nanmin(F['X'][:, j]) >= 0]
def logpos(A):
    A = A.copy(); A[:, POSITIVE] = np.log1p(A[:, POSITIVE]); return A
class E:  # same KRR factory as evaluate.py
    @staticmethod
    def krr():
        return GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), FunctionTransformer(logpos), StandardScaler(), KernelRidge(kernel='rbf')),
                            {'kernelridge__alpha': [0.03, 0.1, 0.3, 1.0], 'kernelridge__gamma': [0.005, 0.01, 0.02, 0.05]},
                            cv=KFold(4, shuffle=True, random_state=0), scoring='neg_mean_squared_error')
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'),
          '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'),
          '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
out, per_id = {}, {}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = d['ids'], d['y'], d['fold']
    p021 = d['fixed_control'] if name == 'original' else d['best021']
    X = F['X'][[pos[i] for i in ids]]
    nobj = np.nan_to_num(X[:, keys.index('n_obj')], nan=0)
    failed = nobj < 20
    g = np.zeros(len(y)); b = np.zeros(len(y)); ws = []
    for k in range(5):
        tr, va = fold != k, fold == k
        mu = y[tr].mean(); m = E.krr().fit(X[tr], y[tr] - mu)
        # inner OOF on the training part to choose the blend weight without touching fold k
        inner = np.zeros(tr.sum()); ftr = fold[tr]; ytr = y[tr]; Xtr = X[tr]
        for j in np.unique(ftr):
            a, v = ftr != j, ftr == j
            mu2 = ytr[a].mean(); inner[v] = E.krr().fit(Xtr[a], ytr[a] - mu2).predict(Xtr[v]) + mu2
        ptr = p021[tr]; inner = np.where(failed[tr], ptr, inner)
        grid = np.linspace(0, 1, 101)
        w = grid[np.argmin([np.mean(((1 - a) * ptr + a * inner - ytr) ** 2) for a in grid])]; ws.append(float(w))
        g[va] = m.predict(X[va]) + mu
        g[va] = np.where(failed[va], p021[va], g[va])
        b[va] = (1 - w) * p021[va] + w * g[va]
    rm = lambda a: float(np.sqrt(np.mean((a - y) ** 2)))
    fold_gain = [float(np.sqrt(np.mean((p021[fold == k] - y[fold == k]) ** 2)) - np.sqrt(np.mean((b[fold == k] - y[fold == k]) ** 2))) for k in range(5)]
    out[name] = dict(best021=rm(p021), grain_krr=rm(g), blend=rm(b), gain=rm(p021) - rm(b), weights=ws, folds_improved=int(sum(x > 0 for x in fold_gain)), n_failed=int(failed.sum()))
    per_id[name] = dict(ids=ids, y=y, p021=p021, blend=b)
    print(name, json.dumps(out[name]), flush=True)
# pooled ID-level bootstrap across the three partitions (same 400 IDs; NOT independent evidence)
ref = per_id['original']['ids']; idx = {n: {i: k for k, i in enumerate(v['ids'])} for n, v in per_id.items()}
se0 = np.mean([(per_id[n]['p021'][[idx[n][i] for i in ref]] - per_id[n]['y'][[idx[n][i] for i in ref]]) ** 2 for n in per_id], 0)
se1 = np.mean([(per_id[n]['blend'][[idx[n][i] for i in ref]] - per_id[n]['y'][[idx[n][i] for i in ref]]) ** 2 for n in per_id], 0)
rng = np.random.default_rng(20261007); gains = []
for _ in range(10000):
    i = rng.integers(0, len(ref), len(ref)); gains.append(np.sqrt(se0[i].mean()) - np.sqrt(se1[i].mean()))
out['pooled'] = dict(best021=float(np.sqrt(se0.mean())), blend=float(np.sqrt(se1.mean())), gain=float(np.sqrt(se0.mean()) - np.sqrt(se1.mean())),
                     ci95=[float(np.quantile(gains, .025)), float(np.quantile(gains, .975))],
                     note='Three partitions of the same 400 images; conditional CI, not independent validation.')
print('pooled', json.dumps(out['pooled']))
json.dump(out, open(_repo_path(R, 'robustness.json'), 'w'), indent=1)
