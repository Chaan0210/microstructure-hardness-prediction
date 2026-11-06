"""Bandwidth path between linear DTA and local learners: log-target kernel ridge (RBF, gamma = c / n_features on standardised features) fitted on
outer-training real rows + degraded copies (deg..deg4, noise >= 12, sources outside the validation fold), variant u1dmavgH2Z. c -> 0 approaches the
linear model. Pilot: standalone noisy-row RMSE per (c, alpha) on dev; any choice must later be fixed a priori and confirmed on holdout."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); v = 'u1dmavgH2Z'
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sid = load(3)['ids']; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
ids_all = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(ids_all)}; XA = F['X'].astype(float)
SX, SID = [], []
for s in ['deg', 'deg2', 'deg3', 'deg4']:
    S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
SX = np.vstack(SX); SID = np.array(SID)
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
grid = [(c, a) for c in (0.003, 0.01, 0.03, 0.1, 0.3) for a in (0.03, 0.3, 3)]
out = {f'c{c}_a{a}': np.zeros((3, 400)) for c, a in grid}
for p_, (name, path) in enumerate(splits.items()):
    d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
    for k in range(5):
        tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vs = set(va); keep = np.array([s not in vs for s in SID])
        X = np.vstack([XA[[pos[i] for i in tr]], SX[keep]]); yy = np.log(np.r_[[lab[i] for i in tr], [lab[s] for s in SID[keep]]])
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(X); sc = StandardScaler().fit(imp.transform(X)); Z = np.nan_to_num(sc.transform(imp.transform(X)))
        Zv = np.nan_to_num(sc.transform(imp.transform(XA[[pos[i] for i in va]]))); mu = yy.mean(); jj = [sid.index(i) for i in va]
        for c, a in grid:
            m = KernelRidge(alpha=a, kernel='rbf', gamma=c / Z.shape[1]).fit(Z, yy - mu); out[f'c{c}_a{a}'][p_, jj] = np.exp(m.predict(Zv) + mu)
        print(name, k, flush=True)
np.savez(_repo_io('krr-dta.npz'), **out)
