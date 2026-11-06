"""Nested check of the interpretable formula on the dev protocol: inside each outer training fold, forward-select OLS terms (candidates as in
genform.py, plus pairwise products of the selected terms) by inner 5-fold CV on the CLEAN training rows, fit, predict the held-out clean rows.
Compares with 054 on the clean dev rows and as a partner on clean rows. usage: genform_dev.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, _repo_io('../push-900'))
from blend_dev import load, rmse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
B = load(9); ids = B['ids']; y = B['y']; FO = B['fold']; c = B['c054']
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; X = F['X'][[pos[i] for i in ids]].astype(float)
P = np.load(_repo_io('/private/tmp/claude-501/-Users-dongchan-Desktop-PythonWorkspace-microstructure-2026/f62e347b-cb1a-4c17-8d54-2bf505be6706/scratchpad/presets.npz')); nzm = dict(zip(P['tr'], P['ntr'])); NG = np.array([nzm[i] for i in ids]); cl = NG < 9.5
cand = ['u_ph', 'm_ph_hard', 'z_f25', 'sh_cor15', 'aw_inv_sqrt_d', 'grain_inv_sqrt_d', 'aw_eqd_mean', 'grain_eqd_mean', 'n_obj', 'pore_frac', 'px_pore_frac', 'ar_wmean',
        'orient_coherence', 'ang_c4', 'ang_s4', 'grain_logA_std', 'het4_u', 'tile_spfrac_std', 'sp_contiguity', 'sp_to_matrix_size', 'sp_ar', 'm_el_pix', 'solidity_wmean',
        'coarse_area_frac', 'fine_area_frac', 'pore_d_mean', 'neighbors_mean', 'sp_z_mean', 'obj_I_cv', 'm_la_mean']
C0 = {k: X[:, keys.index(k)] for k in cand}
def cvr(Z, yy):
    pr = np.zeros(len(yy))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(yy): pr[va] = LinearRegression().fit(Z[tr], yy[tr]).predict(Z[va])
    return float(np.sqrt(np.mean((pr - yy) ** 2)))
def fit_predict(trm, vam):
    C = {}
    for k, v in C0.items(): med = np.nanmedian(v[trm]); C[k] = np.where(np.isfinite(v), v, med)
    yt = y[trm]; sel = []; cur = 1e9
    for _ in range(12):
        best = min(((cvr(np.column_stack([C[q][trm] for q in sel + [k]]), yt), k) for k in C if k not in sel), key=lambda t: t[0])
        if best[0] > cur - 0.02: break
        sel.append(best[1]); cur = best[0]
    for a in list(sel):
        for b in list(sel):
            if a <= b: C[a + '*' + b] = C[a] * C[b]
    for _ in range(6):
        best = min(((cvr(np.column_stack([C[q][trm] for q in sel + [k]]), yt), k) for k in C if k not in sel), key=lambda t: t[0])
        if best[0] > cur - 0.02: break
        sel.append(best[1]); cur = best[0]
    Z = lambda m: np.column_stack([C[q][m] for q in sel]); return LinearRegression().fit(Z(trm), yt).predict(Z(vam)), sel
G = np.full((3, 400), np.nan); sels = []
for k in range(3):
    for f in range(5):
        trm, vam = (FO[k] != f) & cl, (FO[k] == f) & cl
        G[k, vam], s = fit_predict(trm, vam); sels.append(s)
    print('partition', k, 'done', flush=True)
yc = np.broadcast_to(y, G.shape)[:, cl]; Gc = G[:, cl]; cc = c[:, cl]
print('clean dev rows', cl.sum(), ' formula OLS (nested)', round(rmse(Gc, yc[0]), 3), ' 054', round(rmse(cc, yc[0]), 3))
for a in (0.2, 0.3, 0.5):
    print(f'  clean rows: (1-{a})*054 + {a}*formula ->', round(rmse(cc + a * (Gc - cc), yc[0]), 3), ' parts', [round(rmse((cc + a * (Gc - cc))[k], yc[0]), 3) for k in range(3)], ' vs 054 parts', [round(rmse(cc[k], yc[0]), 3) for k in range(3)])
from collections import Counter; print('most selected terms', Counter([t for s in sels for t in s]).most_common(15))
np.savez(_repo_io('genform-dev.npz'), G=G, clean=cl)
