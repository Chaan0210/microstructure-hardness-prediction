"""Label-free: does per-column quantile mapping of the degraded copies onto the real noise>=12 train rows' marginals (fit on half the real rows,
judged on the other half) remove the multivariate copy-vs-real discrimination? No labels read."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
E = _RepoPath(__file__).resolve().parents[4]; v = 'u1dmavgH2Z'
F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float); ids = np.array([str(i) for i in F['ids']])
ni = keys.index('noise'); real = np.nan_to_num(X[:, ni] * 255, nan=0) >= 12
SX, SID = [], []
for s in ['deg', 'deg2', 'deg3', 'deg4']:
    S = np.load(_repo_path(E, f'push-1200/synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += list(np.array(S['ids'])[m])
SX = np.vstack(SX); SID = np.array([str(i) for i in SID]); Xr = X[real]; idr = ids[real]
def qmap(src, ref):   # per-column monotone map of src values onto ref's empirical quantiles (NaNs kept)
    out = src.copy()
    for j in range(src.shape[1]):
        a, r = src[:, j], ref[:, j]; ok, okr = ~np.isnan(a), ~np.isnan(r)
        if ok.sum() < 5 or okr.sum() < 5 or j == ni: continue
        rk = (np.argsort(np.argsort(a[ok])) + 0.5) / ok.sum(); out[ok, j] = np.quantile(r[okr], rk)
    return out
rng = np.random.RandomState(0); half = rng.rand(len(Xr)) < 0.5
SQ = qmap(SX, Xr[half])
def auc(A, Bc, grpA, grpB):
    D = np.vstack([A, Bc]); lab = np.r_[np.zeros(len(A)), np.ones(len(Bc))]; grp = np.r_[grpA, grpB]; cols = [j for j in range(D.shape[1]) if j != ni]; out = {}
    for kind in ('logit', 'gbm'):
        pr = np.zeros(len(lab))
        for tr, va in GroupKFold(5).split(D, lab, grp):
            m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)) if kind == 'logit' else \
                make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0))
            m.fit(D[tr][:, cols], lab[tr]); pr[va] = m.predict_proba(D[va][:, cols])[:, 1]
        out[kind] = round(float(roc_auc_score(lab, pr)), 3)
    return out
print('raw copies vs held-out real half      ', auc(Xr[~half], SX, idr[~half], SID))
print('QM copies  vs held-out real half      ', auc(Xr[~half], SQ, idr[~half], SID))
print('reference: real half vs other real half', auc(Xr[~half], Xr[half], idr[~half], idr[half]))
