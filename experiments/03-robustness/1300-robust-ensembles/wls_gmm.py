"""WLS version of the gmm-Ridge partner (push-900/gmm_partner.py dev protocol: same features, folds, alpha grid by 3-fold inner CV), with
inverse-variance weights from an in-fold variance model e^2 ~ a + b/N + c*1[noise>=12] fitted (NNLS) on inner OOF residuals of the unweighted fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy.optimize import nnls
sys.argv = ['gmm_partner.py', 'none']; src = open(_repo_io('../push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]
G = {'__file__': '../push-900/gmm_partner.py', '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), G)
from sklearn.model_selection import KFold
S9 = G['S9']; base = np.load(_repo_io(S9 / 'base.npz'), allow_pickle=True); ids = [str(i) for i in base['ids']]; y = base['y']
Btr = np.load(_repo_io(S9 / 'object-bank-train.npz'), allow_pickle=True); X = np.c_[base['X98'], G['gmm_feats'](Btr, ids)]
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]
Nn = np.clip(np.nan_to_num(base['X98'][:, keys.index('n_obj')], nan=20), 20, None); nzf = (np.nan_to_num(base['X98'][:, keys.index('noise')] * 255, nan=0) >= 12).astype(float)
D = np.c_[np.ones(len(y)), 1 / Nn, nzf]; ALPHAS = G['ALPHAS']; ridge = G['ridge']
def fit_predict_w(Xtr, ytr, w, Xte, seed):
    sc = []
    for a in ALPHAS:
        err = []
        for itr, iva in KFold(3, shuffle=True, random_state=seed).split(Xtr):
            m = ridge(a).fit(Xtr[itr], ytr[itr], ridge__sample_weight=w[itr]); err.append((w[iva] * (m.predict(Xtr[iva]) - ytr[iva]) ** 2).sum() / w[iva].sum())
        sc.append(np.mean(err))
    a = ALPHAS[int(np.argmin(sc))]; return ridge(a).fit(Xtr, ytr, ridge__sample_weight=w).predict(Xte)
P = np.zeros((3, 400))
for s in range(3):
    for f in range(5):
        tr, va = np.where(base['fold'][s] != f)[0], np.where(base['fold'][s] == f)[0]; seed = 19001 + 101 * s + f
        res = np.zeros(len(tr))
        for a_, b_ in KFold(5, shuffle=True, random_state=f).split(tr):
            res[b_] = y[tr[b_]] - G['fit_predict'](X[tr[a_]], y[tr[a_]], X[tr[b_]], seed)[0]
        cf, _ = nnls(D[tr], res ** 2); cf[0] = max(cf[0], 1.0); w = 1 / (D[tr] @ cf); w /= w.mean()
        P[s, va] = fit_predict_w(X[tr], y[tr], w, X[va], seed)
    print('partition', s, flush=True)
np.savez(_repo_io('wls-gmm.npz'), ids=np.array(ids), oof=P)
O = np.load(_repo_io('../push-900/gmm-partner-dev.npz'))['oof']; print('gmm alone RMSE: unweighted %.3f -> WLS %.3f' % (np.sqrt(((O - y) ** 2).mean()), np.sqrt(((P - y) ** 2).mean())))
