"""GP (RBF+white on quantile-normal features) on official dev partitions/folds; blend with 039 grain OOF at fixed a; W=0.75 pooled."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
variants = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
D = {v: np.load(_repo_io(f'../push-600/dev-{v}-angh.npz'), allow_pickle=True) for v in variants}
parts = ['original', '20261031', '20261101']
GPO = {}
for v in variants:
    Fz = np.load(_repo_io(f'feat-{v}-angh-train.npz'), allow_pickle=True); pos = {str(i): k for k, i in enumerate(Fz['ids'])}; XA = Fz['X'].astype(float)
    for p in parts:
        ids = [str(i) for i in D[v][p + '_ids']]; y = D[v][p + '_y']; fold = D[v][p + '_fold']; X = XA[[pos[i] for i in ids]]; o = np.zeros(len(y))
        for k in range(5):
            tr, va = fold != k, fold == k
            pp = make_pipeline(SimpleImputer(strategy='median'), QuantileTransformer(n_quantiles=200, output_distribution='normal'), StandardScaler()); Z = pp.fit_transform(X[tr]); mu = y[tr].mean()
            gp = GaussianProcessRegressor(C(300.0) * RBF(8.0) + WhiteKernel(60.0), random_state=0).fit(Z, y[tr] - mu); o[va] = gp.predict(pp.transform(X[va])) + mu
        GPO[(v, p)] = o
    print(v, 'done', flush=True)
np.save(_repo_io('dev-gp-angh.npy'), {f'{v}|{p}': GPO[(v, p)] for v, p in GPO}, allow_pickle=True)
for a in (0.0, 0.1, 0.2, 0.3):
    se = []
    ref = [str(i) for i in D[variants[0]]['original_ids']]
    for p in parts:
        ids = [str(i) for i in D[variants[0]][p + '_ids']]; ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]
        y = D[variants[0]][p + '_y'][j]; q = D[variants[0]][p + '_p021'][j]; f = D[variants[0]][p + '_failed'][j]
        g = np.mean([D[v][p + '_g'] for v in variants], 0)[j]; gp = np.mean([GPO[(v, p)] for v in variants], 0)[j]
        gg = (1 - a) * g + a * gp; se.append((0.25 * q + 0.75 * np.where(f, q, gg) - y) ** 2)
    print('a=%.1f pooled W0.75 %.4f  per-part %s  | gp alone grain rmse %s' % (a, np.sqrt(np.mean(se, 0).mean()), ' '.join('%.3f' % np.sqrt(s.mean()) for s in se),
          ' '.join('%.3f' % np.sqrt(np.mean((np.mean([GPO[(v, p)] for v in variants], 0) - D[variants[0]][p + '_y']) ** 2)) for p in parts)))
