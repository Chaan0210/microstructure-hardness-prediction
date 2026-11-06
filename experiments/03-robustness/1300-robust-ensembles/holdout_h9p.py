"""Holdout-100: plain partners (nb columns) with and without the 2 H9 cumulant columns, fit on dev-400, scored on non-gated holdout rows; plus the proxy
blend with the H9 TabPFN block from holdout-h9.npz."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; dev = set(str(i) for i in np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['ids'])
G = json.load(open(_repo_io('abl/groups.json'))); drop = set(G['intensity'].split() + G['het'].split()); V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
Cu = np.load(_repo_io('cumul-train.npz')); cpos = {str(i): j for j, i in enumerate(Cu['ids'])}
src = open(_repo_io('../push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]; gp = {'__file__': '../push-900/gmm_partner.py', '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), gp)
def run(v, kind, h9, gmm=False):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
    X = F['X'].astype(float)[:, [j for j, k in enumerate(keys) if k not in drop]]
    if gmm: X = np.c_[X, gp['gmm_feats'](np.load(_repo_io('../probabilistic-grains-900/object-bank-train.npz'), allow_pickle=True), ids)]
    if h9: X = np.c_[X, Cu['X'][[cpos[i] for i in ids]][:, [1, 4]]]
    tr = np.array([i in dev for i in ids]); y = np.array([lab[i] for i in ids])
    if kind == 'gmm': p = gp['fit_predict'](X[tr], y[tr], X[~tr], 19001)[0]
    else:
        m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(X[tr], np.log(y[tr]) if kind == 'log' else y[tr])
        p = np.exp(m.predict(X[~tr])) if kind == 'log' else m.predict(X[~tr])
    return p, np.array(ids)[~tr], y[~tr]
H = np.load(_repo_io('holdout-featdrop.npz')); H9 = np.load(_repo_io('holdout-h9.npz')); hid = [str(i) for i in H['hid']]; ng = H['noise'] < 12; S = ['038', '039', '044']; Vs = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
tab = np.mean([H9[f'{s}_{v}_nb'] for s in S for v in Vs], 0)
P = {}
for h9 in (False, True):
    r, ids_, yh = run('u1dmavgH2Z', 'ridge', h9); assert list(ids_) == hid
    l = np.exp(np.mean([np.log(run(v, 'log', h9)[0]) for v in V], 0)); gm = run('u1dmavgH2Z', 'gmm', h9, gmm=True)[0]; P[h9] = (r, l, gm)
    print('h9' if h9 else 'base', 'partner RMSE non-gated: ridge %.3f logr %.3f gmm %.3f' % tuple(np.sqrt(((q - yh)[ng] ** 2).mean()) for q in (r, l, gm)))
bl = lambda pr: (0.384 * tab + 0.128 * pr[0] + 0.128 * pr[1] + 0.15 * pr[2]) / 0.79
d = (bl(P[True]) - yh)[ng] ** 2 - (bl(P[False]) - yh)[ng] ** 2; print('holdout proxy blend (H9 TabPFN + partners) non-gated: dMSE %+.2f ± %.2f' % (d.mean(), d.std() / np.sqrt(ng.sum())))
