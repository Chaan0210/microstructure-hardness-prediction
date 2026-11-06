"""D2: adversarial validation train(500) vs test(1000) on the 039 avg-variant features; per-column KS, per noise half."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
E = str(_repo_path(_REPO_ROOT, 'experiments')) + '/'
Tr = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); Te = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-test.npz'), allow_pickle=True)
keys = [str(k) for k in Tr['keys']]
A, B = Tr['X'].astype(float), Te['X'].astype(float)
X = np.r_[A, B]; z = np.r_[np.zeros(len(A)), np.ones(len(B))]
print('nan frac train/test', np.isnan(A).mean().round(4), np.isnan(B).mean().round(4))
def auc(X, z, model):
    o = np.zeros(len(z)); aucs = []
    for rep in range(3):
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, z):
            m = model(); m.fit(X[tr], z[tr]); o[va] = m.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(z, o))
    return np.mean(aucs), o
a1, o = auc(X, z, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15))
Xi = np.nan_to_num(X, nan=np.nanmedian(X, 0)[None, :].repeat(len(X), 0)[np.isnan(X)] if False else 0)
Xi = np.where(np.isnan(X), np.nanmedian(X, 0), X)
a2, _ = auc(Xi, z, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)))
print(f'adversarial AUC: HGB {a1:.3f}  logistic {a2:.3f}')
nz = X[:, keys.index('noise')] * 255
for name, msk in (('clean(noise<11.9)', nz < 11.9), ('noisy', nz >= 11.9)):
    a, _ = auc(X[msk], z[msk], lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15))
    print(name, 'n', msk.sum(), 'train frac', round(1 - z[msk].mean(), 3), 'AUC', round(a, 3))
ks = []
for j, k in enumerate(keys):
    a, b = A[:, j], B[:, j]; a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    s = stats.ks_2samp(a, b); ks.append((s.pvalue, s.statistic, k, a.mean(), b.mean(), a.std(), b.std()))
ks.sort()
print('top KS columns (p, D, col, mean tr, mean te, sd tr, sd te)')
for r in ks[:15]: print(f'{r[0]:.4f} {r[1]:.3f} {r[2]:28s} {r[3]:.4g} {r[4]:.4g} {r[5]:.4g} {r[6]:.4g}')
print('n cols with p<0.01:', sum(r[0] < 0.01 for r in ks), 'expected ~', round(0.01 * len(ks), 1))
