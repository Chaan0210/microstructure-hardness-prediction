"""Label-free pixel-level realism: copy-vs-real-noisy discrimination on the pixstats.py statistics (GBM + logistic, grouped 5-fold CV by source
image), and per-statistic standardized mean differences. Copies and real rows are restricted to measured noise >= 12 as in the feature gate.
usage: pix_gap.py NAME [NAME ...]   (NAME = work/ps/<NAME>.npz)"""
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
W = _repo_path(Path(__file__).resolve().parent, 'ps')
R = np.load(_repo_path(W, 'real.npz')); keys = [str(k) for k in R['keys']]; Xr = R['X']; idr = np.array([str(i) for i in R['ids']]); ni = keys.index('noise')
EXCL = {'noise'}
def load(n):
    S = np.load(_repo_path(W, f'{n}.npz')); X = S['X']; m = X[:, ni] >= 12; return X[m], np.array([str(i) for i in S['ids']])[m]
def gap(SX, src, cols=None, seed=0):
    cols = cols or [j for j, k in enumerate(keys) if k not in EXCL]
    D = np.vstack([Xr, SX]); lab = np.r_[np.zeros(len(Xr)), np.ones(len(SX))]; grp = np.r_[idr, src]; out = {}
    for kind in ('logit', 'gbm'):
        pr = np.zeros(len(lab))
        for tr, va in GroupKFold(5).split(D, lab, grp):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)) if kind == 'logit' else \
                HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
            m.fit(D[tr][:, cols], lab[tr]); pr[va] = m.predict_proba(D[va][:, cols])[:, 1]
        out[kind] = round(float(roc_auc_score(lab, pr)), 4)
    sd = np.sqrt(0.5 * (Xr.var(0) + SX.var(0))) + 1e-12; smd = (SX.mean(0) - Xr.mean(0)) / sd
    out['smd'] = {keys[j]: round(float(smd[j]), 2) for j in np.argsort(-np.abs(smd)) if keys[j] not in EXCL}
    out['n'] = int(len(SX)); return out
if __name__ == '__main__':
    res = {}
    for n in sys.argv[1:]:
        if n.startswith('POOL:'):
            L = [load(t) for t in n[5:].split(',')]; SX = np.vstack([a for a, _ in L]); src = np.concatenate([b for _, b in L])
        else: SX, src = load(n)
        res[n] = gap(SX, src); print(f"{n:28s} n {res[n]['n']:4d} logit {res[n]['logit']:.3f} gbm {res[n]['gbm']:.3f}  top SMD", list(res[n]['smd'].items())[:6], flush=True)
    json.dump(res, open(_repo_path(W, f'gap-{"_".join(a.replace(":", "").replace(",", "+") for a in sys.argv[1:])[:80]}.json'), 'w'), indent=1)
