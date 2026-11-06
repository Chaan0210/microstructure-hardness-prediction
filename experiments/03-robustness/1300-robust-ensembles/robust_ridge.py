"""Linear partners with robust per-column transforms fitted in-fold (no column selection): quantile->normal, winsorise 1/99%, Yeo-Johnson.
RidgeCV (u1dmavgH2Z) and logRidge alpha 2 (4 variants, geometric mean), plain + DTA d1234 (noise >= 12), Study900 dev folds -> blend as in 061.
usage: robust_ridge.py MODE   (MODE: none|qn|win|yj)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import RidgeCV, Ridge
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
MODE = sys.argv[1]; V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
class Win(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): self.lo, self.hi = np.nanpercentile(X, 1, 0), np.nanpercentile(X, 99, 0); return self
    def transform(self, X): return np.clip(X, self.lo, self.hi)
def tf():
    return {'none': [], 'qn': [QuantileTransformer(n_quantiles=200, output_distribution='normal', subsample=10**9, random_state=0)], 'win': [Win()],
            'yj': [PowerTransformer('yeo-johnson', standardize=False)]}[MODE]
def mk(kind): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), *tf(), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
def oof(v, kind, dta):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
    pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float); SX, SID = [], []
    if dta:
        for s in ['deg', 'deg2', 'deg3', 'deg4']:
            S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
        SX = np.vstack(SX)
    SID = np.array(SID); O = np.zeros((3, 400))
    for p_, path in enumerate(splits.values()):
        d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va)
            X = XA[[pos[i] for i in tr]]; yy = np.array([lab[i] for i in tr])
            if dta: kp = np.array([s not in vs for s in SID]); X = np.vstack([X, SX[kp]]); yy = np.r_[yy, [lab[s] for s in SID[kp]]]
            m = mk(kind).fit(X, np.log(yy) if kind == 'log' else yy); pr = m.predict(XA[[pos[i] for i in va]])
            O[p_, [sid.index(i) for i in va]] = np.exp(pr) if kind == 'log' else pr
    return O
out = {}
for dta in (0, 1):
    out[f'ridge_{dta}'] = oof('u1dmavgH2Z', 'ridge', dta); out[f'logridge_{dta}'] = np.exp(np.mean([np.log(oof(v, 'log', dta)) for v in V], 0)); print(MODE, 'dta', dta, 'done', flush=True)
np.savez(_repo_io(f'rr-{MODE}.npz'), **out)
