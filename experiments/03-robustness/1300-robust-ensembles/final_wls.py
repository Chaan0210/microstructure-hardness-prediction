"""Final (500 train rows) WLS partners for candidate 070: RidgeCV (u1dmavgH2Z) and logRidge alpha 2 (4 variants), plain (all real rows) and DTA
(+ copies deg..deg4 with measured noise >= 12), and the gmm-Ridge partner, with inverse-variance weights from a variance model e^2 ~ a + b/N + c*noisy
fitted (NNLS) on 5-fold OOF residuals of the unweighted fit on the 500 rows (copies of inner-validation sources excluded). WEIGHTED=0 reproduces
the production predictions (fin-none / fin-d1234n12 / gmm-partner-test) as a check. Nothing fitted on test. usage: final_wls.py [0|1]"""
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
WEIGHTED = (sys.argv[1] if len(sys.argv) > 1 else '1') == '1'
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
order = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/sample_submission.csv')))]; V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
def var_design(X, keys):
    return np.c_[np.ones(len(X)), 1 / np.clip(np.nan_to_num(X[:, keys.index('n_obj')], nan=20), 20, None), (np.nan_to_num(X[:, keys.index('noise')] * 255, nan=0) >= 12).astype(float)]
def mk(kind): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True) if kind == 'ridge' else SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
def fit(kind, X, yy, w):
    m = mk(kind); m.fit(X, np.log(yy) if kind == 'log' else yy, **{m.steps[-1][0] + '__sample_weight': w}); return m
def prd(kind, m, X): p = m.predict(X); return np.exp(p) if kind == 'log' else p
def partner(v, kind, dta):
    Tr = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); Te = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-test.npz'), allow_pickle=True)
    keys = [str(k) for k in Tr['keys']]; ni = keys.index('noise'); assert [str(i) for i in Te['ids']] == order
    ids = np.array([str(i) for i in Tr['ids']]); X = Tr['X'].astype(float); y = np.array([lab[i] for i in ids]); Xe = Te['X'].astype(float)
    SX, SID = [], []
    if dta:
        for s in ['deg', 'deg2', 'deg3', 'deg4']:
            S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
        SX = np.vstack(SX)
    else: SX = np.zeros((0, X.shape[1]))
    SID = np.array(SID); SY = np.array([lab[s] for s in SID])
    if not WEIGHTED: return prd(kind, fit(kind, np.vstack([X, SX]), np.r_[y, SY], np.ones(len(y) + len(SY))), Xe)
    res = np.zeros(len(y))
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):
        bs = set(ids[b]); kk = np.array([s not in bs for s in SID], bool)
        res[b] = y[b] - prd(kind, fit(kind, np.vstack([X[a], SX[kk]]), np.r_[y[a], SY[kk]], np.ones(len(a) + kk.sum())), X[b])
    D = var_design(X, keys); cf, _ = nnls(D, res ** 2); cf[0] = max(cf[0], 1.0)
    w = np.r_[1 / (D @ cf), 1 / (var_design(SX, keys) @ cf) if len(SX) else []]; w /= w.mean(); print(v, kind, 'dta' if dta else 'plain', 'variance model', np.round(cf, 1), flush=True)
    return prd(kind, fit(kind, np.vstack([X, SX]), np.r_[y, SY], w), Xe)
out = {}
for dta in (0, 1):
    out[f'ridge_{dta}'] = partner('u1dmavgH2Z', 'ridge', dta); out[f'logridge_{dta}'] = np.exp(np.mean([np.log(partner(v, 'log', dta)) for v in V], 0))
# gmm-Ridge (push-900/gmm_partner.py final protocol, alpha by 3-fold CV on the grid) with WLS
src = open(_repo_io('../push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]; G = {'__file__': '../push-900/gmm_partner.py', '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), G)
Ftr = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); Fte = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-test.npz'), allow_pickle=True); keys = [str(k) for k in Ftr['keys']]
tid = [str(i) for i in Ftr['ids']]; eid = [str(i) for i in Fte['ids']]; S9 = G['S9']
Xtr = np.c_[Ftr['X'].astype(float), G['gmm_feats'](np.load(_repo_io(S9 / 'object-bank-train.npz'), allow_pickle=True), tid)]; Xte = np.c_[Fte['X'].astype(float), G['gmm_feats'](np.load(_repo_io(S9 / 'object-bank-test.npz'), allow_pickle=True), eid)]
ytr = np.array([lab[i] for i in tid])
if not WEIGHTED: out['gmm'] = G['fit_predict'](Xtr, ytr, Xte, 19001)[0]
else:
    res = np.zeros(len(ytr))
    for a, b in KFold(5, shuffle=True, random_state=0).split(Xtr): res[b] = ytr[b] - G['fit_predict'](Xtr[a], ytr[a], Xtr[b], 19001)[0]
    D = var_design(Ftr['X'].astype(float), keys); cf, _ = nnls(D, res ** 2); cf[0] = max(cf[0], 1.0); w = 1 / (D @ cf); w /= w.mean(); print('gmm variance model', np.round(cf, 1))
    sc = []
    for al in G['ALPHAS']:
        err = []
        for itr, iva in KFold(3, shuffle=True, random_state=19001).split(Xtr):
            m = G['ridge'](al).fit(Xtr[itr], ytr[itr], ridge__sample_weight=w[itr]); err.append((w[iva] * (m.predict(Xtr[iva]) - ytr[iva]) ** 2).sum() / w[iva].sum())
        sc.append(np.mean(err))
    out['gmm'] = G['ridge'](G['ALPHAS'][int(np.argmin(sc))]).fit(Xtr, ytr, ridge__sample_weight=w).predict(Xte)
np.savez(_repo_io(f'fin-wls{int(WEIGHTED)}.npz'), ids=np.array(order), **out); print('saved', f'fin-wls{int(WEIGHTED)}.npz')
if not WEIGHTED:
    N0 = np.load(_repo_path(H12, 'fin-none.npz')); N1 = np.load(_repo_path(H12, 'fin-d1234n12.npz')); Gm = np.load(_repo_io('../push-900/gmm-partner-test.npz'))
    print('reproduction max |diff|: ridge0', np.abs(out['ridge_0'] - N0['ridge']).max(), 'logr0', np.abs(out['logridge_0'] - N0['logridge']).max(),
          'ridge1', np.abs(out['ridge_1'] - N1['ridge']).max(), 'logr1', np.abs(out['logridge_1'] - N1['logridge']).max(), 'gmm', np.abs(out['gmm'] - Gm['pred']).max())
