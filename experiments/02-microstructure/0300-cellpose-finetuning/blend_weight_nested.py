"""DEPRECATED (2026-10-09 review) - do not use. The per-fold weight choice below is NOT nested: the OOF predictions on the outer-train
rows (best021 fixed_control and the harness grain OOF from 500-image folds) were produced by models trained with the outer-validation
labels. Candidates are compared at the fixed grain weight W=0.75 (dev_combine.py).

Nested choice of the best021 vs grain-TabPFN blend weight on the 3 dev partitions (400 IDs, 5 folds each).
grain = harness TabPFN OOF (oof_base_tabpfn.npy, mean of 3 reps, 500-image folds); best021 = fixed_control OOF.
const: w_grain chosen on the outer-train folds from 0..1 (step 0.05); noise: w_grain = clip(a + b*noise_rank) (rejected in push-300)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys; sys.exit('blend_weight_nested.py is deprecated (not nested; see docstring). Use dev_combine.py fixed weights.')
sys.path.insert(0, '.')
import numpy as np
from scipy.stats import rankdata
from harness import *
X, keys, ids = load_feats('features-v3tta'); nz = np.load(_repo_io('noise_train.npy')); pos = {i: k for k, i in enumerate(ids)}
base = np.load(_repo_io(sys.argv[1] if len(sys.argv) > 1 else 'oof_base_tabpfn.npy')).mean(0)
nr = (rankdata(nz) - 0.5) / len(nz)
for part, path in [('original', 'experiments/resolution-216/ensemble-OOF.npz'), ('20261031', 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'),
                   ('20261101', 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]:
    d = np.load(_repo_io(P / path)); ix = np.array([pos[i] for i in d['ids']]); yy = d['y']; p021 = d['fixed_control']; g = base[ix]; r = nr[ix]; fold = d['fold']
    rm = lambda p: np.sqrt(np.mean((p - yy) ** 2))
    res = {'021': rm(p021), 'grain': rm(g), 'w.5': rm(.5 * p021 + .5 * g)}
    grid = [(a, b) for a in np.linspace(0.3, 1, 15) for b in np.linspace(-1, 0.2, 13)]
    out = np.zeros(len(yy)); out_c = np.zeros(len(yy)); wcs = []
    for k in range(5):
        tr, va = fold != k, fold == k
        loss = lambda a, b, s: np.mean((np.clip(a + b * r[s], 0, 1) * g[s] + (1 - np.clip(a + b * r[s], 0, 1)) * p021[s] - yy[s]) ** 2)
        a, b = min(grid, key=lambda t: loss(*t, tr)); w = np.clip(a + b * r[va], 0, 1); out[va] = w * g[va] + (1 - w) * p021[va]
        wc = min(np.linspace(0, 1, 21), key=lambda w: np.mean((w * g[tr] + (1 - w) * p021[tr] - yy[tr]) ** 2)); out_c[va] = wc * g[va] + (1 - wc) * p021[va]; wcs.append(round(float(wc), 2))
    res['nested_const_w'] = rm(out_c); res['nested_noise_w'] = rm(out)
    print(part, {k: round(float(v), 3) for k, v in res.items()}, 'chosen const w_grain per fold', wcs)
