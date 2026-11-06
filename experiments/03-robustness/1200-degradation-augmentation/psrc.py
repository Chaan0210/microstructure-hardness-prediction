"""Paired-simulation regression calibration (label-free): the degraded copies of clean train images give exact pairs
(features measured on the degraded copy, features measured on the clean original). A multi-output RidgeCV M: x_noisy -> x_clean is fitted on
these pairs only (no hardness labels, no real noisy rows, no test rows) and applied to every real row (train and test) with pixel noise >= 12;
the 'noise' column is kept as measured. -> push-1200/feat-<v>-anghcorRC-<split>.npz (044-slot feature sets).
usage: psrc.py SETS(comma)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sets = sys.argv[1].split(',')
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
for v in V:
    Tr = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in Tr['keys']]; pos = {str(i): k for k, i in enumerate(Tr['ids'])}
    XN, XC = [], []
    for s in sets:
        S = np.load(_repo_path(H, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); XN.append(S['X'].astype(float)); XC.append(Tr['X'][[pos[str(i)] for i in S['ids']]].astype(float))
    XN, XC = np.vstack(XN), np.vstack(XC); j_n = keys.index('noise'); out_cols = [j for j in range(len(keys)) if j != j_n]
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(XN); sc = StandardScaler().fit(imp.transform(XN))
    impC = SimpleImputer(strategy='median', keep_empty_features=True).fit(XC[:, out_cols])
    M = RidgeCV(alphas=np.logspace(-2, 3, 11), alpha_per_target=True).fit(sc.transform(imp.transform(XN)), impC.transform(XC[:, out_cols]))
    r2 = 1 - ((M.predict(sc.transform(imp.transform(XN))) - impC.transform(XC[:, out_cols])) ** 2).mean(0) / impC.transform(XC[:, out_cols]).var(0).clip(1e-12)
    for split in ('train', 'test'):
        F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-{split}.npz'), allow_pickle=True); X = F['X'].astype(float).copy()
        g = np.nan_to_num(X[:, j_n] * 255, nan=0) >= 12
        X[np.ix_(g, out_cols)] = M.predict(sc.transform(imp.transform(X[g])))
        np.savez(_repo_path(H, f'feat-{v}-anghcorRC-{split}.npz'), ids=F['ids'], X=X, keys=F['keys'])
    print(v, 'pairs', len(XN), 'in-sample R2 median %.3f, key cols: ' % np.median(r2) + ' '.join(f'{keys[j]} {r2[i]:.2f}' for i, j in enumerate(out_cols) if keys[j] in ('u_ph', 'het4_u', 'm_la_mean', 'sh_cor15', 'z_f25', 'sizefield_std_s32')), flush=True)
