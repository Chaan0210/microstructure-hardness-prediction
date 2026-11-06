"""Diagnostic for the block heterogeneity idea: Spearman and partial Spearman of het metrics with the 030 harness OOF residual by noise tercile."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from scipy.stats import spearmanr, rankdata
from harness import labels
hetf = sys.argv[1] if len(sys.argv) > 1 else 'het-H300-train.npz'
Z = np.load(_repo_io(hetf), allow_pickle=True); hk = [str(k) for k in Z['keys']]; ids = [str(i) for i in Z['ids']]; y = labels(ids)
F = np.load(_repo_io('feat-a1s250n4vstd-stcoh-train.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; X = F['X']
O = np.mean([np.load(_repo_io(f'oof_none_v2_feat-a1s250n4vstd-stcoh-train_s{s}.npy')).mean(0) for s in (0, 1, 2)], 0)
r = y - O; nz = np.load(_repo_io('noise_train.npy')); t = np.digitize(nz, np.quantile(nz, [1 / 3, 2 / 3]))
ctrl = ['tile_logA_std', 'tile8_logA_std', 'sizefield_std_s32', 'n_obj']
C = np.column_stack([X[:, fk.index(k)] for k in ctrl] + [O])
def pcorr(a, b, C):
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(C).all(1)
    Cm = np.c_[np.ones(ok.sum()), np.column_stack([rankdata(c) for c in C[ok].T])]
    res = lambda v: rankdata(v) - Cm @ np.linalg.lstsq(Cm, rankdata(v), rcond=None)[0]
    return float(np.corrcoef(res(a[ok]), res(b[ok]))[0, 1])
print('residual = y - 030 harness OOF (v2 n8, mean of seeds 0,1,2); RMSE by tercile', [round(float(np.sqrt(np.mean(r[t == j] ** 2))), 3) for j in range(3)])
print(f"{'metric':13s} | rho(resid) clean mid noisy | partial rho (ctrl {'/'.join(ctrl)}, pred) clean mid noisy | rho(y) clean | rho with tile8_logA_std (clean)")
for k in hk:
    v = Z['X'][:, hk.index(k)]
    a = [spearmanr(v[t == j], r[t == j], nan_policy='omit').correlation for j in range(3)]
    b = [pcorr(v[t == j], r[t == j], C[t == j]) for j in range(3)]
    c0 = spearmanr(v[t == 0], X[t == 0, fk.index('tile8_logA_std')], nan_policy='omit').correlation
    print(f'{k:13s} | ' + ' '.join(f'{x:6.3f}' for x in a) + ' | ' + ' '.join(f'{x:6.3f}' for x in b) + f' | {spearmanr(v[t == 0], y[t == 0], nan_policy="omit").correlation:6.3f} | {c0:6.3f}')
m = (t == 0) & np.isfinite(Z['X']).all(1); A = np.c_[np.ones(m.sum()), Z['X'][m, hk.index('het4_b')], Z['X'][m, hk.index('neff_q')]]
coef = np.linalg.lstsq(A, r[m], rcond=None)[0]; fit = A @ coef
print('clean-tercile in-sample OLS resid ~ 1 + het4 + neff^0.25: coef', np.round(coef, 3), 'R2', round(1 - np.var(r[m] - fit) / np.var(r[m]), 4), 'n', int(m.sum()))
# significance reference: |rho| for n~165 at p=0.05 is ~0.153
