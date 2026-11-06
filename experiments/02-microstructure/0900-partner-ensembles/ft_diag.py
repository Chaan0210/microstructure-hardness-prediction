"""Label-free check of fine-tuned Cellpose segmenters on the real noisy half (train): structure is independent of the preset, so the noisy-half
feature distribution of an unbiased segmenter matches the clean half (zero-shot masks-v3 features). Shift in clean-SD units and SD ratio.
usage: ft_diag.py NAME[,NAME...]   (push-300/ftseg-ft-<NAME>-train-noisy-n1.npz, view 0)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
E = _RepoPath(__file__).resolve().parents[2]
F = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz')); k = [str(x) for x in F['keys']]; X = F['X']; nzv = X[:, k.index('noise')]; thr = np.nanmedian(nzv)
nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255; ids = [str(i) for i in F['ids']]
COLS = ['n_obj', 'grain_mean_area_j', 'grain_eqd_mean', 'aw_eqd_mean', 'grain_logA_std', 'coverage', 'sp_frac' if 'sp_frac' in k else 'pore_frac', 'ar_wmean', 'neighbors_mean']
COLS = [c for c in COLS if c in k]; clean = nz < 9.5
for name in sys.argv[1].split(','):
    D = np.load(_repo_path(E, f'push-300/ftseg-ft-{name}-train-noisy-n1.npz'), allow_pickle=True); dk = [str(x) for x in D['keys']]; did = [str(i) for i in D['ids']]
    hi = np.array([nz[ids.index(i)] >= 12 for i in did])
    out = []
    for c in COLS:
        ref = X[clean, k.index(c)]; ref = ref[np.isfinite(ref)]; v = D['X'][hi, 0, dk.index(c)]; v = v[np.isfinite(v)]
        out.append(f'{c} {(np.median(v) - np.median(ref)) / ref.std():+.2f}/{v.std() / ref.std():.2f}')
    print(f'{name:8s} (noise>=12 rows, median shift / SD ratio vs clean) ' + '  '.join(out))
