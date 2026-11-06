from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from scipy.stats import spearmanr
from harness import load_feats, labels, tabpfn
X, keys, ids = load_feats('features-v3'); y = labels(ids)
D = np.load(_repo_io('fidelity.npz'), allow_pickle=True); src = D['src']; dk = [str(k) for k in D['keys']]; assert dk == keys
variants = [v for v in ['deg', 'nlm', 'up2', 'nlm_up2'] if v in D.files]
key_f = ['sp_frac_grains', 'sp_count_frac', 'f_dark_20', 'sizefield_range_s32', 'tile8_logA_std', 'grain_eqd_mean', 'aw_eqd_mean', 'grain_logA_std', 'ar_wmean', 'orient_coherence', 'solidity_wmean', 'px_pore_n', 'n_obj']
C = D['clean']
print(f"{'feature':22s} " + ' '.join(f'{v:>8s}' for v in variants))
for k in key_f:
    j = keys.index(k); row = []
    for v in variants:
        a, b = C[:, j], D[v][:, j]; ok = np.isfinite(a) & np.isfinite(b)
        row.append(spearmanr(a[ok], b[ok]).correlation if ok.sum() > 10 else np.nan)
    print(f'{k:22s} ' + ' '.join(f'{r:8.3f}' for r in row))
rest = np.setdiff1d(np.arange(len(y)), src)
P = {}
for sd in (0, 1):
    m = tabpfn(8, sd, 'mps'); m.fit(X[rest], y[rest])
    for v in ['clean'] + variants: P.setdefault(v, []).append(m.predict(D[v]))
for v in ['clean'] + variants:
    p = np.mean(P[v], 0); print(f'{v:8s} RMSE on 80 sources: {np.sqrt(np.mean((p - y[src]) ** 2)):.3f}')
