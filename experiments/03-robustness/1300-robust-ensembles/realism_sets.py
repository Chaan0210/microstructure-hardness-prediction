"""Feature-space domain AUC per copy set vs real noisy train rows (98 production cols; logistic C 0.1 and HGB, 5-fold CV), plus the top
noise-adjusted standardized mean differences for the given sets. usage (from push-1200): realism_sets.py SET [SET ...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; XA = F['X'].astype(float)
ni = keys.index('noise'); nzA = np.nan_to_num(XA[:, ni] * 255, nan=0); R = XA[nzA >= 12]
drop = [keys.index(k) for k in keys if k == 'noise']; keep = [j for j in range(len(keys)) if j not in drop]   # noise itself excluded (copies' noise is matched by design)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for s in sys.argv[1:]:
    S = np.load(_repo_io(f'synthrows-{s}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); X = S['X'].astype(float); X = X[X[:, ni] * 255 >= 12]
    Z = np.vstack([R, X])[:, keep]; d = np.r_[np.ones(len(R)), np.zeros(len(X))]; out = []
    for nm, m in [('logit', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))),
                  ('hgb', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight='balanced'))]:
        out.append(roc_auc_score(d, cross_val_predict(m, Z, d, cv=cv, method='predict_proba')[:, 1]))
    sm = []
    for j, k in enumerate([keys[j] for j in keep]):
        v = Z[:, j]; ok = np.isfinite(v)
        if ok.sum() < 50 or np.nanstd(v) == 0: continue
        r = v[ok]; dd = d[ok]; sm.append(((r[dd == 1].mean() - r[dd == 0].mean()) / r.std(), k))
    sm.sort(key=lambda t: -abs(t[0]))
    print(f'{s:6s} n {len(X):3d} domain AUC logit {out[0]:.3f} hgb {out[1]:.3f} | top SMD ' + ', '.join(f'{k} {v:+.2f}' for v, k in sm[:5]), flush=True)
