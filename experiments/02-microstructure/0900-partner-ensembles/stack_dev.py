"""Honest stacking check on dev: blend weights over the six components (021, grain045, ridge, logRidge, set, gmm) fitted on the OOF rows of the
other four folds of the same partition and applied to the held-out fold (sum-to-one least squares, optional non-negativity), either global or per
noise segment. Compares with the fixed 054 weights. usage: stack_dev.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy.optimize import nnls
from blend_dev import load, rmse, parts
B = load(); y, F, fail = B['y'], B['fold'], B['fail']
names = ['p021', 'g', 'ridge', 'logridge', 'set', 'gmm']; C = np.stack([B[n] for n in names], -1)   # [3, 400, 6]
ids = B['ids']; nz_all = np.load(_repo_io('../push-300/noise_train.npy')); import sys; sys.path.insert(0, _repo_io('../push-300')); import harness
tid = harness.ids('train') if hasattr(harness, 'ids') else None
def fitw(Xc, yc, nonneg):   # min ||Xc w - yc|| s.t. sum w = 1 (substitute last weight), optional w >= 0 via nnls on augmented system
    if nonneg:
        lam = 1e4; A = np.vstack([Xc, lam * np.ones(Xc.shape[1])]); b = np.r_[yc, lam]; w, _ = nnls(A, b); return w
    Z = Xc[:, :-1] - Xc[:, -1:]; t = yc - Xc[:, -1]; v = np.linalg.lstsq(Z, t, rcond=None)[0]; return np.r_[v, 1 - v.sum()]
def run(seg, nonneg):
    P = np.zeros_like(y[None].repeat(3, 0)); W = []
    for k in range(3):
        for f in range(5):
            tr, va = (F[k] != f) & ~fail[k], F[k] == f
            for s in np.unique(seg):
                m_tr, m_va = tr & (seg == s), va & (seg == s)
                w = fitw(C[k][m_tr], y[m_tr], nonneg); W.append(w); P[k, m_va] = C[k][m_va] @ w
    P = np.where(fail, B['p021'], P); return P, np.array(W)
import csv
NZ = None
try:
    from pathlib import Path
    trids = [r['ID'] for r in csv.DictReader(open(_repo_io(Path(_repo_io('../../data/train.csv')))))]; nzmap = dict(zip(trids, nz_all)); NZ = np.array([nzmap[i] for i in ids])
except Exception as e: print('noise map failed', e)
c = B['c054']; print('fixed 054', round(rmse(c, y), 4), parts(c, y).round(3))
segs = {'global': np.zeros(400, int)}
if NZ is not None:
    q = np.quantile(NZ, [1 / 3, 2 / 3]); segs['noise2'] = (NZ > np.median(NZ)).astype(int); segs['noise3'] = np.digitize(NZ, q)
for sn, seg in segs.items():
    for nonneg in (True, False):
        P, W = run(seg, nonneg)
        print(f'{sn:7s} nonneg={nonneg!s:5s}: {rmse(P, y):.4f} ({rmse(c, y) - rmse(P, y):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in parts(c, y) - parts(P, y)) + ')  mean w ' + ' '.join(f'{n}={v:.2f}' for n, v in zip(names, W.mean(0))))
