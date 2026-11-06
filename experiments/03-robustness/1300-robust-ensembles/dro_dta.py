"""Group-DRO DTA partners (reviewer suggestion 1c): groups = real outer-training rows + each degradation set (deg, deg2, deg3, deg4); exponentiated-
gradient group weights q_g <- q_g * exp(eta * L_g / mean L) over 15 rounds (L_g = in-sample weighted MSE of group g, log target for logRidge),
each copy weight = q_g / n_g * n_copies (real rows weight 1 * q_real * n_groups). Same production partner settings otherwise."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); ETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; SETS = ['deg', 'deg2', 'deg3', 'deg4']
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
def oof(v, kind):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
    ids_all = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(ids_all)}; XA = F['X'].astype(float); SX, SID, SG = [], [], []
    for gi, s in enumerate(SETS):
        S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12
        SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]; SG += [gi + 1] * int(m.sum())
    SX = np.vstack(SX); SID = np.array(SID); SG = np.array(SG); O = np.zeros((3, 400)); Q = []
    for p_, (name, path) in enumerate(splits.items()):
        d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va); keep = np.array([s not in vs for s in SID])
            X = np.vstack([XA[[pos[i] for i in tr]], SX[keep]]); yy = np.r_[[lab[i] for i in tr], [lab[s] for s in SID[keep]]]; grp = np.r_[np.zeros(len(tr), int), SG[keep]]
            yt = np.log(yy) if kind == 'log' else yy; G = 1 + len(SETS); q = np.ones(G) / G; nG = np.bincount(grp, minlength=G)
            for it in range(15):
                sw = q[grp] / nG[grp] * len(grp)
                m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
                m.fit(X, yt, **{m.steps[-1][0] + '__sample_weight': sw}); r2 = (m.predict(X) - yt) ** 2; L = np.array([r2[grp == gg].mean() for gg in range(G)])
                q = q * np.exp(ETA * L / L.mean()); q /= q.sum()
            Q.append(q); pr = m.predict(XA[[pos[i] for i in va]]); O[p_, [sid.index(i) for i in va]] = np.exp(pr) if kind == 'log' else pr
    return O, np.mean(Q, 0)
r, q = oof('u1dmavgH2Z', 'ridge'); print('final group weights (real, deg, deg2, deg3, deg4)', np.round(q, 3))
lr = np.exp(np.mean([np.log(oof(v, 'log')[0]) for v in V], 0)); np.savez(_repo_io(f'dro-eta{ETA}.npz'), ridge=r, logridge=lr)
