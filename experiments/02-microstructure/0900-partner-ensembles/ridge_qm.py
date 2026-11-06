"""Ridge partners (Study900 RidgeCV on the average variant; log-target Ridge alpha 2, 4-variant geometric mean) on original vs aligned (qm) features,
dev protocol; evaluated by swapping them into 054. usage: ridge_qm.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import warnings, numpy as np
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from blend_dev import load, rmse, parts
B = load(9); ids, y, FO = B['ids'], B['y'], B['fold']
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
def X_of(path): F = np.load(_repo_io(path), allow_pickle=True); pos = {str(i): k for k, i in enumerate(F['ids'])}; return F['X'][[pos[i] for i in ids]].astype(float)
def fit(XV, Xavg):
    r = np.zeros((3, 400)); lr = np.zeros((3, 400))
    for k in range(3):
        for f in range(5):
            tr, va = FO[k] != f, FO[k] == f
            r[k, va] = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(Xavg[tr], y[tr]).predict(Xavg[va])
            lr[k, va] = np.exp(np.mean([make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(XV[v][tr], np.log(y[tr])).predict(XV[v][va]) for v in V], 0))
    return r, lr
o = {v: X_of(f'../push-800/feat-{v}-anghcor-train.npz') for v in V}; q = {v: X_of(f'feat-{v}-anghcorqm-train.npz') for v in V}
r0, l0 = fit(o, o['u1dmavgH2Z']); r1, l1 = fit(q, q['u1dmavgH2Z'])
print('check vs stored: ridge', np.abs(r0 - B['ridge']).max().round(4), ' logridge', np.abs(l0 - B['logridge']).max().round(4))
np.savez(_repo_io('ridge-qm.npz'), ridge=r1, logridge=l1)
fail, p = B['fail'], B['p021']
def c54(r, l): c51 = 0.2 * p + 0.48 * B['g'] + 0.16 * r + 0.16 * l; return np.where(fail, p, 0.77 * c51 + 0.08 * B['set'] + 0.15 * B['gmm'])
base = c54(r0, l0)
for nm, (r, l) in (('orig', (r0, l0)), ('ridge qm', (r1, l0)), ('logridge qm', (r0, l1)), ('both qm', (r1, l1))):
    c = c54(r, l); print(f'{nm:12s}: ridge alone {rmse(np.where(fail, p, r), y):.3f} logridge alone {rmse(np.where(fail, p, l), y):.3f}  054 {rmse(c, y):.4f} ({rmse(base, y) - rmse(c, y):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in parts(base, y) - parts(c, y)) + ')')
