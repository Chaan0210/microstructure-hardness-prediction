"""Feature-space realism of the degraded copies: real train rows with pixel noise >= 12 vs copies with measured noise >= 12 (98-col production
features, u1dmavgH2Z-anghcor). Domain-classifier CV AUC (logistic, gradient boosting) and per-feature standardized mean differences after
regressing each feature on noise within the pooled set. Also clean vs noisy real label distribution. usage: realism_check.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
lab = {r.split(',')[0]: float(r.split(',')[1]) for r in open(_repo_io('../../data/train.csv')).read().split()[1:]}
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; XA = F['X'].astype(float); ids = [str(i) for i in F['ids']]
ni = keys.index('noise'); nzA = np.nan_to_num(XA[:, ni] * 255, nan=0); yA = np.array([lab[i] for i in ids])
print(f'real: clean(<9.5) n={np.sum(nzA < 9.5)} mean y {yA[nzA < 9.5].mean():.2f} sd {yA[nzA < 9.5].std():.2f} | noisy(>=12) n={np.sum(nzA >= 12)} mean y {yA[nzA >= 12].mean():.2f} sd {yA[nzA >= 12].std():.2f}')
SX, SS = [], []
for s in 'deg,deg2,deg3,deg4,deg5,deg6,deg7'.split(','):
    S = np.load(_repo_io(f'synthrows-{s}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); X = S['X'].astype(float); m = X[:, ni] * 255 >= 12; SX.append(X[m]); SS += [s] * int(m.sum())
SX = np.vstack(SX); SS = np.array(SS); R = XA[nzA >= 12]
print(f'copies {len(SX)} real noisy {len(R)}; noise mean copies {np.nanmean(SX[:, ni] * 255):.2f} real {np.nanmean(R[:, ni] * 255):.2f}')
X = np.vstack([R, SX]); d = np.r_[np.ones(len(R)), np.zeros(len(SX))]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for nm, m in [('logit', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))),
              ('hgb', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight='balanced'))]:
    p = cross_val_predict(m, X, d, cv=cv, method='predict_proba')[:, 1]; print(f'domain AUC {nm}: {roc_auc_score(d, p):.3f}')
    if nm == 'hgb':
        for s in np.unique(SS): mm = np.r_[np.ones(len(R), bool), SS == s]; print(f'   vs {s}: AUC {roc_auc_score(d[mm], p[mm]):.3f}')
# per-feature standardized differences, noise-adjusted (pooled linear fit on noise)
nzp = np.nan_to_num(X[:, ni] * 255, nan=12)
out = []
for j, k in enumerate(keys):
    v = X[:, j]; ok = np.isfinite(v)
    if ok.sum() < 50 or np.nanstd(v) == 0: continue
    A = np.c_[np.ones(ok.sum()), nzp[ok]]; b = np.linalg.lstsq(A, v[ok], rcond=None)[0]; r = v[ok] - A @ b; dd = d[ok]
    out.append(((r[dd == 1].mean() - r[dd == 0].mean()) / r.std(), k, np.isnan(R[:, j]).mean(), np.isnan(SX[:, j]).mean()))
out.sort(key=lambda t: -abs(t[0]))
print('largest noise-adjusted SMD (real - copies), nan-frac real/copies:')
for t in out[:20]: print(f'  {t[1]:28s} {t[0]:+.3f}  nan {t[2]:.2f}/{t[3]:.2f}')
