"""Noisy-row (pixel noise >= 12) blend with the sim-augmented linear partners: simplex-constrained least squares (NNLS, sum 1) over components,
honest evaluation by repeated half-splits of the 400 ids (fit on one half's noisy rows of all 3 partitions, score the other half).
Clean/mid rows keep 053. usage: noisy_stack.py SIMTAG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy.optimize import nnls
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
B = load(3); y = B['y']; fail = B['fail']; nz = np.nan_to_num(np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['noise'], nan=0)
gate = (nz >= 12) & ~fail[0]; S = np.load(_repo_io(f'sr-A-{sys.argv[1]}.npz'))
c53 = np.where(fail, B['p021'], 0.16 * B['p021'] + 0.384 * B['g'] + 0.128 * B['ridge'] + 0.128 * B['logridge'] + 0.05 * B['set'] + 0.15 * B['gmm'])
C = {'p021': B['p021'], 'g': B['g'], 'gmm': B['gmm'], 'set': B['set'], 'ridge_aug': S['ridge'], 'logr_aug': S['logridge']}
names = list(C); M = np.stack([C[n] for n in names], -1)   # [3, 400, K]
def fitw(rows):
    A = M[:, rows].reshape(-1, len(names)); b = y[rows][None].repeat(3, 0).reshape(-1)
    # sum-to-one via heavy penalty row
    A2 = np.vstack([A, 1e3 * np.ones(len(names))]); b2 = np.r_[b, 1e3]; w, _ = nnls(A2, b2); return w
def apply(w): return np.where(gate[None] & ~fail, M @ w, c53)
w_all = fitw(np.where(gate)[0]); print('NNLS weights (all noisy rows):', dict(zip(names, np.round(w_all, 3))))
a = apply(w_all); print('in-sample dev %.4f noisy %.3f parts %s' % (np.sqrt(((a - y) ** 2).mean()), np.sqrt(((a - y)[:, gate] ** 2).mean()), np.round(np.sqrt(((a - y) ** 2).mean(1)), 4)))
rng = np.random.RandomState(0); G = []; W = []
for rep in range(300):
    idx = rng.permutation(400); A_, B_ = idx[:200], idx[200:]; w = fitw(A_[gate[A_]]); W.append(w)
    G.append(((c53[:, B_] - y[B_]) ** 2).mean() - ((apply(w)[:, B_] - y[B_]) ** 2).mean())
G = np.array(G); W = np.array(W)
print('honest dMSE vs 053: mean %.3f sd %.3f P>0 %.2f  (dRMSE approx %.4f)' % (G.mean(), G.std(), (G > 0).mean(), np.sqrt((c53 - y) ** 2).mean() * 0 + 9.7665 - np.sqrt(9.7665 ** 2 - G.mean())))
print('weight spread across halves (mean / sd):', {n: (round(float(W[:, k].mean()), 3), round(float(W[:, k].std()), 3)) for k, n in enumerate(names)})
