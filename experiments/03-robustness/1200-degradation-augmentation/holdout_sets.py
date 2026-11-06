"""Holdout-100 check of partner variants (061 vs 062 ingredients): copy sets {d1234, d1to7} x alphas {production (RidgeCV 0.1..1000, logRidge a2),
internal CV (wide grids)}; fit on 400 dev rows + copies of dev sources with noise >= 12. usage: holdout_sets.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np, warnings
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
warnings.filterwarnings('ignore')
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
dev = set(str(i) for i in np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['ids'])
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
def run(v, sets, wide, model):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; X = F['X'].astype(float); y = np.array([lab[i] for i in ids])
    ni = [str(k) for k in F['keys']].index('noise'); tr = np.array([i in dev for i in ids]); ho = ~tr; Xt, yt = [X[tr]], list(y[tr])
    for s in sets:
        S = np.load(_repo_io(f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); k = np.array([str(i) in dev for i in S['ids']]) & (S['X'][:, ni] * 255 >= 12)
        Xt.append(S['X'][k].astype(float)); yt += [lab[str(i)] for i in np.array(S['ids'])[k]]
    Xt, yt = np.vstack(Xt), np.array(yt)
    if model == 'ridge': p = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000] if wide else [0.1, 1, 10, 100, 1000])).fit(Xt, yt).predict(X[ho])
    else: p = np.exp(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=[0.01, 0.1, 0.3, 1, 2, 3, 10, 30, 100]) if wide else Ridge(alpha=2.0)).fit(Xt, np.log(yt)).predict(X[ho]))
    return p, y[ho], np.nan_to_num(X[ho][:, ni] * 255, nan=0)
for name, sets in (('none', []), ('d1234', 'deg,deg2,deg3,deg4'.split(',')), ('d1to7', 'deg,deg2,deg3,deg4,deg5,deg6,deg7'.split(','))):
    for wide in ((False,) if name == 'none' else (False, True)):
        pr, y, nz = run('u1dmavgH2Z', sets, wide, 'ridge'); pl = np.exp(np.mean([np.log(run(v, sets, wide, 'log')[0]) for v in V], 0)); g = nz >= 12; m = (pr + pl) / 2
        print(f'{name:6s} {"wideCV" if wide else "prod  "}: noisy n={g.sum()} ridge {np.sqrt(((pr - y)[g] ** 2).mean()):.3f} logridge {np.sqrt(((pl - y)[g] ** 2).mean()):.3f} mean {np.sqrt(((m - y)[g] ** 2).mean()):.3f}')
