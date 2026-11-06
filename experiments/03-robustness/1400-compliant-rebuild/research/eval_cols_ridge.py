"""Quick linear information check of extra columns: RidgeCV / logRidge dev OOF (Study900 3 partitions x 5 folds) on feat-u1dmavgH2Z-anghcor with and
without the extra columns; reports per band dMSE with a paired SE, plus a random-column control (same count). Diagnostic only.
usage: eval_cols_ridge.py EXTRA.npz [cols]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
E = str(_repo_path(_REPO_ROOT, 'experiments'))
Z = np.load(_repo_io(sys.argv[1]), allow_pickle=True); keys = [str(k) for k in Z['keys']]; sel = sys.argv[2].split(',') if len(sys.argv) > 2 else keys; ci = [keys.index(c) for c in sel]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io(E + '/../data/train.csv')))}
F = np.load(_repo_io(E + '/push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]; X = F['X'].astype(float)
zid = [str(i) for i in Z['train_ids']]; Xn = Z['train'][[zid.index(i) for i in ids]][:, ci]; nz = np.nan_to_num(X[:, fk.index('noise')] * 255, nan=0)
sid = [str(i) for i in np.load(_repo_io(E + '/probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]; pos = {i: k for k, i in enumerate(ids)}
splits = [E + '/resolution-216/ensemble-OOF.npz', E + '/resolution-repartition-217/20261031/ensemble-OOF.npz', E + '/resolution-repartition-217/20261101/ensemble-OOF.npz']
y = np.array([lab[i] for i in sid]); nzd = nz[[pos[i] for i in sid]]
def dev(Xa, kind):
    O = np.zeros((3, 400))
    for p_, path in enumerate(splits):
        d = np.load(_repo_io(path)); di, fold = [str(i) for i in d['ids']], d['fold']
        for k in range(5):
            tr = [pos[i] for i, f in zip(di, fold) if f != k]; va = [i for i, f in zip(di, fold) if f == k]; yt = np.array([lab[ids[j]] for j in tr])
            m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]) if kind == 'ridge' else Ridge(alpha=2.0))
            m.fit(Xa[tr], np.log(yt) if kind == 'log' else yt); p = m.predict(Xa[[pos[i] for i in va]]); O[p_, [sid.index(i) for i in va]] = np.exp(p) if kind == 'log' else p
    return O
rng = np.random.RandomState(7)
for kind in ('ridge', 'log'):
    b = dev(X, kind); n = dev(np.c_[X, Xn], kind); r = dev(np.c_[X, rng.randn(len(X), len(ci))], kind)
    for nm, m in (('all', np.ones(400, bool)), ('clean<9.5', nzd < 9.5), ('mid', (nzd >= 9.5) & (nzd < 12)), ('noisy>=12', nzd >= 12)):
        d = ((n - y) ** 2 - (b - y) ** 2)[:, m].mean(0); dr = ((r - y) ** 2 - (b - y) ** 2)[:, m].mean(0)
        print(f'{kind:5s} {nm:10s} base RMSE {np.sqrt(((b - y) ** 2)[:, m].mean()):.3f} dMSE new {d.mean():+.2f} (SE {d.std() / np.sqrt(m.sum()):.2f}) | random {dr.mean():+.2f}')
