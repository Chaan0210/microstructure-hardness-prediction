"""Fidelity of a fine-tuned segmenter on the 80 cleanest train images (held out of fine-tuning): features from the clean input and from the
fidelity_exp degradation (blur 1.0 / noise 17 / contrast 0.85) vs. zero-shot features (fidelity.npz). TabPFN fit on the other 420 real rows.
usage: eval_fid.py ftseg-<tag>-train-fid-n2.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.path.insert(0, '.')
import numpy as np
from scipy.stats import spearmanr
from harness import load_feats, labels, tabpfn
X, keys, ids = load_feats('features-v3'); y = labels(ids)
D = np.load(_repo_io('fidelity.npz'), allow_pickle=True); src = D['src']; assert [str(k) for k in D['keys']] == keys
E = np.load(_repo_io(sys.argv[1]), allow_pickle=True); assert [str(i) for i in E['ids']] == [ids[k] for k in src]
V = {'zs_clean': D['clean'], 'zs_deg': D['deg'], 'ft_clean': E['X'][:, 0], 'ft_deg': E['X'][:, 1]}
key_f = ['sp_frac_grains', 'sp_count_frac', 'f_dark_20', 'sizefield_range_s32', 'sizefield_std_s32', 'tile8_logA_std', 'grain_eqd_mean', 'aw_eqd_mean', 'grain_logA_std', 'ar_wmean', 'orient_coherence', 'solidity_wmean', 'neighbors_std', 'n_obj']
C = D['clean']
print(f"{'spearman vs zs_clean':22s} " + ' '.join(f'{v:>9s}' for v in V))
for k in key_f:
    j = keys.index(k); row = []
    for v in V:
        a, b = C[:, j], V[v][:, j]; ok = np.isfinite(a) & np.isfinite(b)
        row.append(spearmanr(a[ok], b[ok]).correlation if ok.sum() > 10 else np.nan)
    print(f'{k:22s} ' + ' '.join(f'{r:9.3f}' for r in row))
j = keys.index('n_obj'); print('median n_obj', {v: float(np.nanmedian(V[v][:, j])) for v in V})
rest = np.setdiff1d(np.arange(len(y)), src); Pr = {}
for sd in (0, 1):
    m = tabpfn(8, sd, 'cpu'); m.fit(X[rest], y[rest])
    for v in V: Pr.setdefault(v, []).append(m.predict(V[v]))
for v in V:
    p = np.mean(Pr[v], 0); print(f'{v:9s} RMSE on 80 sources: {np.sqrt(np.mean((p - y[src]) ** 2)):.3f}')
