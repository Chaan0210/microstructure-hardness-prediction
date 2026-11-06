"""Label-free estimator choice: recover the known second-phase fraction on synthetic Voronoi images with realistic degradations."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys, warnings, json
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from phase_features import grain_table, phase_estimates
from features_v3 import extract_v3
rng = np.random.default_rng(7); yy, xx = np.mgrid[0:256, 0:256]
rows = []
for it in range(160):
    n = int(rng.uniform(60, 600)); st = rng.uniform(1, 3); th = rng.uniform(0, np.pi)
    pts = rng.uniform(-40, 296, (n, 2)); c, s_ = np.cos(th), np.sin(th)
    def tr(p): return np.c_[(c * p[:, 0] + s_ * p[:, 1]) / st, -s_ * p[:, 0] + c * p[:, 1]]
    lab = cKDTree(tr(pts)).query(tr(np.c_[yy.ravel(), xx.ravel()]))[1].reshape(256, 256) + 1
    lab = np.unique(lab, return_inverse=True)[1].reshape(256, 256) + 1; ng = lab.max()
    f_true = rng.uniform(0, 0.5); is_sp = rng.random(ng) < f_true
    lev = np.where(is_sp, rng.normal(0.52, 0.03, ng), rng.normal(0.70, 0.03, ng)); im = lev[lab - 1].copy()
    npore = int(rng.integers(2, 15))
    for _ in range(npore):
        cy, cx = rng.uniform(8, 248, 2); r = rng.uniform(3, 6); m = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r; lab[m] = lab.max() + 1; im[m] = 0.15
    bnd = (ndi.maximum_filter(lab, 2) != ndi.minimum_filter(lab, 2)); im[bnd] -= 0.06
    im = ndi.gaussian_filter(im, rng.uniform(0, 2)); cst = rng.uniform(0.4, 1.5); im = (im - 0.6) * cst + 0.55
    im = im + rng.normal(0, rng.uniform(0.005, 0.12), im.shape)
    u8 = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    # truth on non-pore grains (count fraction among grains present)
    present = np.unique(lab[lab <= ng]); fc_true = float(is_sp[present - 1].mean())
    area_true = float(np.isin(lab, np.where(is_sp)[0] + 1).sum() / (lab <= ng).sum())
    t = grain_table(u8, lab); e = phase_estimates(t); old = extract_v3(u8, lab)
    rows.append(dict(fc_true=fc_true, area_true=area_true, noise=float(np.std(u8 / 255. - ndi.gaussian_filter(u8 / 255., 3))), old_count=old['sp_count_frac'], old_area=old['sp_frac_grains'], **e))
keys = [k for k in rows[0] if k.startswith(('ph_', 'old_'))]
res = {}
for k in keys:
    v = np.array([r[k] for r in rows]); tgt = np.array([r['area_true' if 'area' in k else 'fc_true'] for r in rows])
    res[k] = dict(corr=float(np.corrcoef(v, tgt)[0, 1]))
nz = np.array([r['noise'] for r in rows]); hi = nz > np.median(nz)
for k in keys:
    v = np.array([r[k] for r in rows]); tgt = np.array([r['area_true' if 'area' in k else 'fc_true'] for r in rows])
    res[k]['corr_noisy_half'] = float(np.corrcoef(v[hi], tgt[hi])[0, 1])
for k, r in sorted(res.items(), key=lambda kv: -kv[1]['corr'] if np.isfinite(kv[1]['corr']) else 0):
    print(f"{k:22s} corr {r['corr']:.3f}  noisy-half {r['corr_noisy_half']:.3f}")
json.dump(res, open(_repo_io('phase-synth-test.json'), 'w'), indent=1)
