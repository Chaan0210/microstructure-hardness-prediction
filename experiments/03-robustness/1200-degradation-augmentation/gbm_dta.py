"""Flexible-but-global learner with degraded copies: HistGradientBoosting on real fold-train rows + copies (d1234, noise >= 12, sources not in
the validation fold), vs the same without copies and vs the DTA RidgeCV/logRidge (sr-A-d1234n12). Standalone noisy-row RMSE, Study900 dev.
usage: gbm_dta.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, warnings
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
P = Path(_repo_io('../..')); lab = {r.split(',')[0]: float(r.split(',')[1]) for r in open(_repo_path(P, 'data/train.csv')).read().split()[1:]}
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float); ni = keys.index('noise')
SX, SY, SID = [], [], []
for s in 'deg,deg2,deg3,deg4'.split(','):
    S = np.load(_repo_io(f'synthrows-{s}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); X = S['X'].astype(float); k = X[:, ni] * 255 >= 12
    SX.append(X[k]); SY += [lab[str(i)] for i in np.array(S['ids'])[k]]; SID += [str(i) for i in np.array(S['ids'])[k]]
SX = np.vstack(SX); SY = np.array(SY); SID = np.array(SID)
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
sid = [str(i) for i in np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]; out = {o: np.zeros((3, 400)) for o in ('plain', 'dta')}
mk = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=0)
for p_, (name, path) in enumerate(splits.items()):
    d = np.load(_repo_io(path)); ids = [str(i) for i in d['ids']]; y = d['y']; fold = d['fold']; X = XA[[pos[i] for i in ids]]; o = {q: np.zeros(400) for q in out}
    for k in range(5):
        tr, va = fold != k, fold == k; vids = set(np.array(ids)[va]); keep = np.array([s not in vids for s in SID])
        o['plain'][va] = mk().fit(X[tr], y[tr]).predict(X[va]); o['dta'][va] = mk().fit(np.vstack([X[tr], SX[keep]]), np.r_[y[tr], SY[keep]]).predict(X[va])
    pp = {i: j for j, i in enumerate(ids)}; jj = [pp[i] for i in sid]
    for q in out: out[q][p_] = o[q][jj]
D = np.load(_repo_io('dev065.npz')); yy, nz, fail = D['y'], D['nz'], D['fail']; m = (nz >= 12) & ~fail; S = np.load(_repo_io('sr-A-d1234n12.npz'))
r = lambda a, mm: np.sqrt(((a - yy)[:, mm] ** 2).mean())
for q, a in [('GBM plain', out['plain']), ('GBM + copies', out['dta']), ('RidgeCV + copies (061)', S['ridge']), ('logRidge + copies (061)', S['logridge']), ('TabPFN block (053)', D['g'])]:
    print(f'{q:24s} noisy {r(a, m):.3f}  clean+mid {r(a, (nz < 12) & ~fail):.3f}')
c = D['c']; print('corr of noisy-row errors GBM+copies vs logRidge+copies', np.corrcoef((out['dta'] - yy)[:, m].ravel(), (S['logridge'] - yy)[:, m].ravel())[0, 1].round(3))
np.savez(_repo_io('gbm-dta.npz'), **out)
