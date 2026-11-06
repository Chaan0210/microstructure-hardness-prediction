"""Label-free baseline of the gate metric: domain_gap.py's grouped-CV copy-vs-real-noisy discrimination (GBM / logistic, all 97 non-noise
anghcor columns, u1dmavgH2Z) for each existing copy set separately and for given pooled groups. No labels are read.
usage: gap_sets.py [SET ...]   (SET = push-1200 synthrows name, or path to a synthrows npz)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
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
W = Path(__file__).resolve().parents[1]; E = _RepoPath(__file__).resolve().parents[3]; v = 'u1dmavgH2Z'
F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float); ids = np.array([str(i) for i in F['ids']])
ni = keys.index('noise'); real = np.nan_to_num(X[:, ni] * 255, nan=0) >= 12; Xr, idr = X[real], ids[real]; cols = [j for j in range(len(keys)) if j != ni]
def load(s):
    p = Path(s) if s.endswith('.npz') else _repo_path(E, f'push-1200/synthrows-{s}-{v}-anghcor.npz')
    S = np.load(_repo_io(p), allow_pickle=True); assert [str(k) for k in S['keys']] == keys
    m = S['X'][:, ni].astype(float) * 255 >= 12; return S['X'].astype(float)[m], np.array([str(i) for i in S['ids']])[m]
def gap(SX, src, seed=0):
    D = np.vstack([Xr, SX]); lab = np.r_[np.zeros(len(Xr)), np.ones(len(SX))]; grp = np.r_[idr, src]; out = {}
    for kind in ('logit', 'gbm'):
        pr = np.zeros(len(lab))
        for tr, va in GroupKFold(5).split(D, lab, grp):
            m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)) if kind == 'logit' else \
                make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed))
            m.fit(D[tr][:, cols], lab[tr]); pr[va] = m.predict_proba(D[va][:, cols])[:, 1]
        out[kind] = round(float(roc_auc_score(lab, pr)), 4)
    a1 = []
    for j in cols:
        a, b = Xr[:, j], SX[:, j]; a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 10 or len(b) < 10: continue
        a1.append((round(float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])), 3), keys[j]))
    out['top_cols'] = sorted(a1, key=lambda t: -abs(t[0] - 0.5))[:8]; out['n_copies'] = int(len(SX)); return out
if __name__ == '__main__':
    sets = sys.argv[1:] or ['deg', 'deg2', 'deg3', 'deg4', 'deg5', 'deg6', 'deg7', 'deg8', 'deg9', 'deg10', 'deg11', 'deg12', 'train_degU1']
    res = {}
    for s in sets + ['POOL:deg,deg2,deg3,deg4']:
        if s.startswith('POOL:'):
            L = [load(t) for t in s[5:].split(',')]; SX = np.vstack([a for a, _ in L]); src = np.concatenate([b for _, b in L])
        else:
            try: SX, src = load(s)
            except FileNotFoundError: print('missing', s); continue
        res[s] = gap(SX, src); print(s, res[s]['n_copies'], 'logit', res[s]['logit'], 'gbm', res[s]['gbm'], res[s]['top_cols'][:4], flush=True)
    json.dump(res, open(_repo_io(_repo_path(W, 'work/gap_sets.json') if len(sys.argv) == 1 else _repo_path(W, f'work/gap_{Path(sets[0]).stem}.json')), 'w'), indent=1)
