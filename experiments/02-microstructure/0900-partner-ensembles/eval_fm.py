"""Tabular foundation-model partners on top of 054 (dev): for each model with all 4 variant files dev-<v>-anghcor-<M>.npz present,
slot mean over variants -> standalone RMSE, error correlation with 054 and with the TabPFN-3.5 grain block, (1-a)*054 + a*M for a few a,
and 'grain block +M as a 4th slot' (g' = (3*g + M)/4). Also a joint fit over all available models (dev-optimal shares, for information only).
usage: eval_fm.py"""
import numpy as np
from pathlib import Path
from blend_dev import load, rmse, parts, partner, align_dev
B = load(9); y, fail, p = B['y'], B['fail'], B['p021']; c = B['c054']
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
MODELS = ['v3.5-fast', 'v2.6', 'v3', 'v2.5', 'v2.5-real', 'v2.5-small', 'tabicl', 'tabdpt']
M = {}
for m in MODELS:
    fs = [Path(f'dev-{v}-anghcor-{m}.npz') for v in V]
    if all(f.exists() for f in fs): M[m] = np.mean([align_dev(f) for f in fs], 0)
g = np.where(fail, p, B['g'])
for m, o in M.items():
    oo = np.where(fail, p, o)
    print(f'== {m}: alone {rmse(oo, y):.3f}  err corr with 054 {np.corrcoef((oo - y).ravel(), (c - y).ravel())[0, 1]:.3f}  with grain block {np.corrcoef((oo - y).ravel(), (g - y).ravel())[0, 1]:.3f}')
    for a in (0.25, 0.5):
        gq = (1 - a) * B['g'] + a * o; cq = np.where(fail, p, 0.77 * (0.2 * p + 0.48 * gq + 0.16 * B['ridge'] + 0.16 * B['logridge']) + 0.08 * B['set'] + 0.15 * B['gmm'])
        print(f'   grain block -> {1 - a:.2f} g + {a:.2f} {m}: {rmse(cq, y):.4f} ({rmse(c, y) - rmse(cq, y):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in parts(c, y) - parts(cq, y)) + ')')
    d = np.where(fail, 0, o - B['g']); e = np.where(fail, 0, c - y); a_opt = -np.mean(0.77 * 0.48 * d * e) / np.mean((0.77 * 0.48 * d) ** 2); print(f'   dev-optimal share of the grain block: {a_opt:.2f}')
if len(M) > 1:
    names = list(M); D = np.array([(np.where(fail, p, M[m]) - c).ravel() for m in names]); e = (c - y).ravel()
    w = -np.linalg.solve(D @ D.T, D @ e); cj = c + np.tensordot(w, np.array([np.where(fail, p, M[m]) - c for m in names]), 1)
    print('joint dev-optimal extra shares', dict(zip(names, w.round(3))), '->', round(rmse(cj, y), 4), (parts(c, y) - parts(cj, y)).round(3))
