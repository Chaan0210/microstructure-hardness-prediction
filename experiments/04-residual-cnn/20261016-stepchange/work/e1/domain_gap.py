"""E1 step 1 (label-free): per-column and multivariate domain discrimination between label-preserving degraded copies (deg..deg4, measured
noise >= 12) and REAL train rows with noise >= 12, on the production anghcor feature set (U-Net variant given as argv[1]). No labels are read.
Outputs work/e1/domain-<variant>.json: per-column AUC, multivariate grouped-CV AUC for all columns and for nested invariant subsets."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
E = _RepoPath(__file__).resolve().parents[4]; v = sys.argv[1] if len(sys.argv) > 1 else 'u1dmavgH2Z'
F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float); ids = [str(i) for i in F['ids']]
ni = keys.index('noise'); nz = np.nan_to_num(X[:, ni] * 255, nan=0); real = nz >= 12
SX, SID = [], []
for s in ['deg', 'deg2', 'deg3', 'deg4']:
    S = np.load(_repo_path(E, f'push-1200/synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); assert [str(k) for k in S['keys']] == keys
    m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [f'{s}:{i}' for i in np.array(S['ids'])[m]]
SX = np.vstack(SX); src = np.array([i.split(':')[1] for i in SID])
Xr = X[real]; idr = np.array(ids)[real]
D = np.vstack([Xr, SX]); lab = np.r_[np.zeros(len(Xr)), np.ones(len(SX))]; grp = np.r_[idr, src]
cols = [j for j in range(len(keys)) if j != ni]
auc1 = {}
for j in cols:
    a, b = Xr[:, j], SX[:, j]; ok_a, ok_b = ~np.isnan(a), ~np.isnan(b)
    if ok_a.sum() < 10 or ok_b.sum() < 10 or np.nanstd(np.r_[a[ok_a], b[ok_b]]) == 0: auc1[keys[j]] = 0.5; continue
    sc = np.r_[a[ok_a], b[ok_b]]; ll = np.r_[np.zeros(ok_a.sum()), np.ones(ok_b.sum())]; auc1[keys[j]] = float(roc_auc_score(ll, sc))
dev = {k: abs(a - 0.5) for k, a in auc1.items()}
def multi(cs, kind):
    pr = np.zeros(len(lab))
    for tr, va in GroupKFold(5).split(D, lab, grp):
        m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)) if kind == 'logit' else \
            make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0))
        m.fit(D[tr][:, cs], lab[tr]); pr[va] = m.predict_proba(D[va][:, cs])[:, 1]
    return float(roc_auc_score(lab, pr))
res = dict(variant=v, n_real=int(len(Xr)), n_copies=int(len(SX)), per_column_auc=dict(sorted(auc1.items(), key=lambda t: -abs(t[1] - 0.5))))
res['multi_all'] = dict(logit=multi(cols, 'logit'), gbm=multi(cols, 'gbm'))
for tau in [0.10, 0.075, 0.05]:
    cs = [j for j in cols if dev[keys[j]] <= tau]
    res[f'multi_dev<={tau}'] = dict(n_cols=len(cs), logit=multi(cs, 'logit'), gbm=multi(cs, 'gbm'), cols=[keys[j] for j in cs])
json.dump(res, open(_repo_path(Path(__file__).parent, f'domain-{v}.json'), 'w'), indent=1)
print('n_real', len(Xr), 'n_copies', len(SX)); print('top gap columns', list(res['per_column_auc'].items())[:15])
for k in res:
    if k.startswith('multi'): print(k, {a: b for a, b in res[k].items() if a != 'cols'})
