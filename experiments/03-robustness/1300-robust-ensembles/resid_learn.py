"""Residual learnability of the 073 dev blend: can a second-stage learner on the 98 features (+ the blend prediction) predict the OOF residual?
Nested: for each partition, 10-fold CV over the band's rows (the residuals are already out-of-fold). Learners: RidgeCV, GBM (shallow), kNN-15.
Permutation null (rows' features shuffled) for the CV R^2."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
import nsgain as N
sid = load(3)['ids']; F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); pos = {str(i): k for k, i in enumerate(F['ids'])}; X = F['X'].astype(float)[[pos[i] for i in sid]]
c73 = np.load(_repo_io('dev-nb-candidate.npy')); y, nz, fail = N.y, N.nz, N.fail; f = fail[0]
L = {'ridge': lambda: make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11))),
     'gbm': lambda: HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=150, min_samples_leaf=20),
     'knn': lambda: make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsRegressor(15))}
def cvr2(Xb, r, mk, seed=0):
    pr = np.zeros(len(r))
    for a, b in KFold(10, shuffle=True, random_state=seed).split(Xb): pr[b] = mk().fit(Xb[a], r[a]).predict(Xb[b])
    return 1 - ((r - pr) ** 2).sum() / ((r - r.mean()) ** 2).sum()
rng = np.random.RandomState(0)
for nm, m in (('clean<11', (nz < 11) & ~f), ('noisy>=12', (nz >= 12) & ~f)):
    for ln, mk in L.items():
        r2 = []; nul = []
        for k in range(3):
            Xb = np.c_[X[m], c73[k, m]]; r = (y - c73[k])[m]; r2.append(cvr2(Xb, r, mk, k))
            for t in range(5): nul.append(cvr2(Xb[rng.permutation(len(r))], r, mk, t))
        print(f'{nm:9s} {ln:5s} CV R2 {np.mean(r2):+.4f} (parts {" ".join(f"{x:+.3f}" for x in r2)}) | null mean {np.mean(nul):+.4f} 95% {np.quantile(nul, 0.95):+.4f}', flush=True)
