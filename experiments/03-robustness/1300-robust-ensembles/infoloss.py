"""Information content of the measured features per set: 5-fold CV R^2 of RidgeCV (98 production cols) and correlations of key features with the
label, for real clean rows, real noisy rows and each copy set (copies with measured noise >= 12, label = source label). usage: infoloss.py SET..."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float); y = np.array([lab[str(i)] for i in F['ids']])
ni = keys.index('noise'); nz = np.nan_to_num(X[:, ni] * 255, nan=0)
def info(nm, Z, yy):
    m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
    p = cross_val_predict(m, Z, yy, cv=KFold(5, shuffle=True, random_state=0)); r2 = 1 - ((p - yy) ** 2).mean() / yy.var()
    c = {k: np.corrcoef(np.nan_to_num(Z[:, keys.index(k)], nan=np.nanmedian(Z[:, keys.index(k)])), yy)[0, 1] for k in ('u_ph', 'z_f25', 'sh_cor15', 'het4_u', 'm_la_mean')}
    print(f'{nm:14s} n {len(yy):3d} CV R2 {r2:.3f} RMSE {np.sqrt(((p - yy) ** 2).mean()):.2f} | corr ' + ' '.join(f'{k} {v:+.2f}' for k, v in c.items()), flush=True)
cl = nz < np.median(nz); info('real clean(src)', X[cl], y[cl]); info('real noisy>=12', X[nz >= 12], y[nz >= 12])
for s in sys.argv[1:]:
    S = np.load(_repo_io(f'synthrows-{s}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); Z = S['X'].astype(float); m = Z[:, ni] * 255 >= 12; info(s, Z[m], np.array([lab[str(i)] for i in np.array(S['ids'])[m]]))
