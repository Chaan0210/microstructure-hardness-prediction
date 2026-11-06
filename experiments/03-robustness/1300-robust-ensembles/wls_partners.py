"""Heteroscedastic (WLS) Ridge partners. Variance model per outer fold, from inner 5-fold OOF residuals of the unweighted partner on the
outer-training real rows: e^2 ~ a + b/N + c*1[noise>=12] (NNLS); weights 1/sigma^2 for real rows; copies (DTA) get the noisy-row variance
a + b/N_source_copy + c. Partners: RidgeCV (u1dmavgH2Z) and logRidge alpha 2 (4 variants), plain (all rows) and DTA (d1234 copies)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.optimize import nnls
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); POW = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
def mk(kind): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
def fit(kind, X, yy, w):
    m = mk(kind); m.fit(X, np.log(yy) if kind == 'log' else yy, **{m.steps[-1][0] + '__sample_weight': w}); return m
def pred(kind, m, X): p = m.predict(X); return np.exp(p) if kind == 'log' else p
def oof(v, kind, dta):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni, nn = keys.index('noise'), keys.index('n_obj')
    ids_all = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(ids_all)}; XA = F['X'].astype(float)
    SX, SID = [], []
    if dta:
        for s in ['deg', 'deg2', 'deg3', 'deg4']:
            S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
        SX = np.vstack(SX)
    SID = np.array(SID); O = np.zeros((3, 400)); coefs = []
    def design(X): Nn = np.clip(np.nan_to_num(X[:, nn], nan=20), 20, None); nz = np.nan_to_num(X[:, ni] * 255, nan=0) >= 12; return np.c_[np.ones(len(X)), 1 / Nn, nz.astype(float)]
    for p_, (name, path) in enumerate(splits.items()):
        d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va); keep = np.array([s not in vs for s in SID], bool)
            Xr = XA[[pos[i] for i in tr]]; yr = np.array([lab[i] for i in tr]); Xc = SX[keep] if dta else np.zeros((0, XA.shape[1])); yc = np.array([lab[s] for s in SID[keep]])
            res = np.zeros(len(tr))   # inner OOF residuals of the unweighted partner (real rows only, copies of inner-val sources excluded)
            for a, b in KFold(5, shuffle=True, random_state=k).split(Xr):
                bset = set(np.array(tr)[b]); kk = np.array([s not in bset for s in SID[keep]], bool) if dta else np.zeros(0, bool)
                Xt = np.vstack([Xr[a], Xc[kk]]) if dta else Xr[a]; yt = np.r_[yr[a], yc[kk]] if dta else yr[a]
                res[b] = yr[b] - pred(kind, fit(kind, Xt, yt, np.ones(len(yt))), Xr[b])
            Dm = design(Xr); cf, _ = nnls(Dm, res ** 2); cf[0] = max(cf[0], 1.0); coefs.append(cf)
            wr = 1 / (Dm @ cf) ** POW
            if dta: Dc = design(Xc); wc = 1 / (Dc @ cf) ** POW; X = np.vstack([Xr, Xc]); yy = np.r_[yr, yc]; w = np.r_[wr, wc]
            else: X, yy, w = Xr, yr, wr
            w = w / w.mean(); O[p_, [sid.index(i) for i in va]] = pred(kind, fit(kind, X, yy, w), XA[[pos[i] for i in va]])
    return O, np.mean(coefs, 0)
out = {}
for dta in (False, True):
    r, cf = oof('u1dmavgH2Z', 'ridge', dta); out[f'ridge_{int(dta)}'] = r; print('dta', dta, 'variance model a, b(/N), c(noisy):', np.round(cf, 1), flush=True)
    out[f'logridge_{int(dta)}'] = np.exp(np.mean([np.log(oof(v, 'log', dta)[0]) for v in V], 0))
np.savez(_repo_io(f'wls-p{POW}.npz'), **out)
