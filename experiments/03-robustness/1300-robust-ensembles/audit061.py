"""061 audit on the official dev (Study900 400 rows x 3 partitions): exact reproduction, error budget by noise band, component errors,
error correlations, and fold-honest simplex stacking (weights fitted on 4 folds' OOF rows, scored on the 5th) as a structure check."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
from scipy.optimize import nnls
B = load(3); y = B['y']; fail = B['fail']; fold = B['fold']
D = np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True); nz = np.nan_to_num(D['noise'], nan=0)
S = np.load(_repo_io('../push-1200/sr-A-d1234n12.npz')); gate = (nz >= 12) & ~fail[0]
C = dict(cnn=B['p021'], tab=B['g'], ridge=np.where(gate, S['ridge'], B['ridge']), logr=np.where(gate, S['logridge'], B['logridge']), set=B['set'], gmm=B['gmm'])
W = dict(cnn=0.16, tab=0.384, ridge=0.128, logr=0.128, set=0.05, gmm=0.15)
c61 = np.where(fail, B['p021'], sum(W[k] * C[k] for k in W))
c53 = np.where(fail, B['p021'], 0.16 * B['p021'] + 0.384 * B['g'] + 0.128 * B['ridge'] + 0.128 * B['logridge'] + 0.05 * B['set'] + 0.15 * B['gmm'])
r = lambda a, m=None: float(np.sqrt(((a - y)[:, m] ** 2).mean())) if m is not None else float(np.sqrt(((a - y) ** 2).mean()))
print('053', round(r(c53), 4), '061', round(r(c61), 4), 'parts', np.round(np.sqrt(((c61 - y) ** 2).mean(1)), 4))
bands = {'clean<9.5': nz < 9.5, 'mid': (nz >= 9.5) & (nz < 12), 'noisy>=12': nz >= 12}
print('fail rows', int(fail[0].sum()), 'n per band', {k: int(m.sum()) for k, m in bands.items()})
tot = ((c61 - y) ** 2).sum()
for k, m in bands.items():
    print(f'{k:10s} 061 {r(c61, m):.3f}  share of SSE {((c61 - y)[:, m] ** 2).sum() / tot:.3f} | ' + ' '.join(f'{c} {r(C[c], m & ~fail[0]):.2f}' for c in C))
# error correlations (noisy and clean)
for k, m in bands.items():
    E = np.array([(C[c] - y)[:, m].ravel() for c in C]); print(k, 'err corr\n', np.round(np.corrcoef(E), 2))
# fold-honest stacking per band (simplex via NNLS on [w>=0], then normalise; no intercept)
def stack(mask_band, comps):
    out = np.zeros((3, 400))
    for p in range(3):
        for k in range(5):
            tr = (fold[p] != k) & mask_band & ~fail[p]; va = (fold[p] == k) & mask_band
            A = np.array([C[c][p, tr] for c in comps]).T; w, _ = nnls(A, y[tr]); w = w / w.sum()
            out[p, va] = np.array([C[c][p, va] for c in comps]).T @ w
    return np.where(fail, B['p021'], out)
comps = list(C)
for k, m in bands.items():
    s = stack(m, comps); print(f'honest stack band {k}: {r(s, m):.3f} vs 061 {r(c61, m):.3f}')
    A = np.array([C[c][:, m & ~fail[0]].ravel() for c in comps]).T; w, _ = nnls(A, np.tile(y[m & ~fail[0]], 3)); print('   in-sample weights', dict(zip(comps, np.round(w / w.sum(), 3))))
allm = np.ones(400, bool); s = stack(allm, comps); print('honest stack global', round(r(s), 4))
sb = c61.copy()
for k, m in bands.items(): sb[:, m] = stack(m, comps)[:, m]
print('honest stack per band', round(r(sb), 4), 'parts', np.round(np.sqrt(((sb - y) ** 2).mean(1)), 4))
np.savez(_repo_io('audit061.npz'), c61=c61, c53=c53, y=y, nz=nz, fail=fail, fold=fold, **{f'C_{k}': v for k, v in C.items()})
