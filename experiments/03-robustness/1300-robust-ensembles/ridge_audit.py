"""061 partner audit: (1) reproduce sr-A-d1234n12 dev OOF; (2) per-source equal total weight for copies; (3) alpha chosen by source-grouped
inner CV instead of GCV (copies of one source never split across inner folds); (4) both; (5) logRidge alpha grid by grouped CV.
All fits use outer-training rows + copies of sources outside the validation fold only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, csv, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import GroupKFold
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200'))
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; SETS = os.environ.get('SETS', 'deg,deg2,deg3,deg4').split(',')
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'),
          '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
AL = [0.1, 1, 10, 100, 1000]; LAL = [0.1, 0.3, 1, 2, 3, 10, 30, 100]
def data(v):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
    ids = [str(i) for i in F['ids']]; SX, SID = [], []
    for s in SETS:
        S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12
        SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
    return ids, F['X'].astype(float), np.vstack(SX), np.array(SID)
def pipe(model, a): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), model)
def gcv_alpha(X, yt, w, grp, grid, log):
    best, ba = np.inf, None
    for a in grid:
        e = 0.0
        for tr, va in GroupKFold(5).split(X, yt, grp):
            m = pipe(Ridge(alpha=a), a).fit(X[tr], yt[tr], ridge__sample_weight=w[tr]); e += (w[va] * (m.predict(X[va]) - yt[va]) ** 2).sum()
        if e < best: best, ba = e, a
    return ba
def oof(v, kind, eqw, grouped):
    ids_all, XA, SX, SID = data(v); pos = {i: k for k, i in enumerate(ids_all)}; O = np.zeros((3, 400)); alphas = []
    cnt = {s: (SID == s).sum() for s in set(SID)}
    for p_, (name, path) in enumerate(splits.items()):
        d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va); keep = np.array([s not in vs for s in SID])
            X = np.vstack([XA[[pos[i] for i in tr]], SX[keep]]); yy = np.r_[[lab[i] for i in tr], [lab[s] for s in SID[keep]]]
            w = np.r_[np.ones(len(tr)), [1.0 / cnt[s] if eqw else 1.0 for s in SID[keep]]]; grp = np.r_[tr, SID[keep]]
            yt = np.log(yy) if kind == 'log' else yy
            if grouped: a = gcv_alpha(X, yt, w, grp, LAL if kind == 'log' else AL, kind == 'log')
            elif kind == 'log': a = 2.0
            else: a = None
            m = pipe(Ridge(alpha=a) if a is not None else RidgeCV(alphas=AL), a).fit(X, yt, **{('ridge' if a is not None else 'ridgecv') + '__sample_weight': w})
            alphas.append(a if a is not None else m[-1].alpha_)
            pr = m.predict(XA[[pos[i] for i in va]]); O[p_, [sid.index(i) for i in va]] = np.exp(pr) if kind == 'log' else pr
    return O, alphas
res = {}
for nm, eqw, grp in [('base', False, False), ('eqw', True, False), ('grp', False, True), ('eqw+grp', True, True)]:
    r, ar = oof('u1dmavgH2Z', 'ridge', eqw, grp); ls = []; la = []
    for v in V:
        o, a = oof(v, 'log', eqw, grp); ls.append(np.log(o)); la += a
    res[nm] = dict(ridge=r, logridge=np.exp(np.mean(ls, 0)))
    print(nm, 'ridge alphas', np.unique(ar, return_counts=True), 'log alphas', np.unique(la, return_counts=True), flush=True)
np.savez(_repo_io('ridge_audit.npz'), **{f'{k}_{c}': v[c] for k, v in res.items() for c in v})
S = np.load(_repo_path(H12, 'sr-A-d1234n12.npz')); print('reproduce base ridge', np.abs(res['base']['ridge'] - S['ridge']).max(), 'logridge', np.abs(res['base']['logridge'] - S['logridge']).max())
