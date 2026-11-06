"""Selection-bias control: 100 random N(0,1) columns; pick the 3 with largest |Spearman| vs the 038 dev residual (pooled over partitions,
same 400 dev rows used to pick mI2); write feat-avgH2Z-nz3-train.npz."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, os
from scipy.stats import spearmanr
D = [np.load(_repo_io(f'../push-300/dev-{t}-v35.npz'), allow_pickle=True) for t in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']]
parts = ['original', '20261031', '20261101']; ref = list(D[0]['original_ids']); G = []
for n in parts:
    ids = list(D[0][f'{n}_ids']); p = D[0][f'{n}_p021']; f = D[0][f'{n}_failed']; g = np.mean([d[f'{n}_g'] for d in D], 0); g = np.where(f, p, g)
    ix = {i: k for k, i in enumerate(ids)}; G.append(g[[ix[i] for i in ref]])
y0 = D[0]['original_y'][[list(D[0]['original_ids']).index(i) for i in ref]]; res = y0 - np.mean(G, 0)
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); fid = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(fid)}
rng = np.random.RandomState(12345); Z = rng.randn(500, 100); j = [pos[i] for i in ref]
r = np.array([spearmanr(Z[j, c], res).correlation for c in range(100)]); top = np.argsort(-np.abs(r))[:3]
print('selected noise |r|', np.round(r[top], 3))
np.savez(_repo_io('feat-avgH2Z-nz3-train.npz'), X=np.c_[F['X'], Z[:, top]], keys=np.array([str(k) for k in F['keys']] + ['nz_a', 'nz_b', 'nz_c']), ids=F['ids'])
