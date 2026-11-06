"""Spline-additive (GAM-like) Ridge partner, dev OOF on the Study900 400 rows x 3 partitions x 5 folds (same folds as the Ridge partners).
Each of the 98 columns -> cubic B-spline basis (quantile knots fitted in-fold) -> RidgeCV (GCV over a grid). plain (real rows) and DTA (+ deg..deg4 copies
with measured noise >= 12; copies of validation-fold sources excluded). usage: spline_dev.py NKNOTS [log]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import RidgeCV
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); NK = int(sys.argv[1]); LOG = len(sys.argv) > 2 and sys.argv[2] == 'log'
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
AL = np.logspace(-1, 4, 16)
def mk(): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), SplineTransformer(n_knots=NK, degree=3, knots='quantile', extrapolation='linear'), RidgeCV(alphas=AL))
def oof(v, dta):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
    ids_all = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(ids_all)}; XA = F['X'].astype(float)
    SX, SID = [], []
    if dta:
        for s in ['deg', 'deg2', 'deg3', 'deg4']:
            S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
        SX = np.vstack(SX)
    SID = np.array(SID); O = np.zeros((3, 400)); al = []
    for p_, (name, path) in enumerate(splits.items()):
        d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va)
            X = XA[[pos[i] for i in tr]]; yy = np.array([lab[i] for i in tr])
            if dta: keep = np.array([s not in vs for s in SID], bool); X = np.vstack([X, SX[keep]]); yy = np.r_[yy, [lab[s] for s in SID[keep]]]
            m = mk().fit(X, np.log(yy) if LOG else yy); p = m.predict(XA[[pos[i] for i in va]]); al.append(m.steps[-1][1].alpha_)
            O[p_, [sid.index(i) for i in va]] = np.exp(p) if LOG else p
    return O, al
out = {}
for dta in (False, True):
    o, al = oof('u1dmavgH2Z', dta); out[f'spl_{int(dta)}'] = o; print('dta', dta, 'alpha median', np.median(al), flush=True)
tag = f'spline-k{NK}{"-log" if LOG else ""}'; np.savez(_repo_io(f'{tag}.npz'), **out); print('saved', tag)
