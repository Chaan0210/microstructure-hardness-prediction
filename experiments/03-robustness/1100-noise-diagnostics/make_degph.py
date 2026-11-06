"""Diagnostic (not a candidate): how much of the noisy-row excess error can phase-measurement error explain?
Clean rows (pixel noise < 9.5) get their phase-type columns degraded to correlation rho with the original: x' = mu + rho (x - mu) + sqrt(1-rho^2) sd e,
with ONE shared standard-normal e per image (real noisy-row phase errors are shared across measures) mixed with 30% independent noise per column.
Optionally the size-type columns too. Noisy/mid rows unchanged. Then TabPFN dev on all rows; compare clean-row RMSE with the undegraded run.
usage: make_degph.py RHO TAG [size]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
rho, tag = float(sys.argv[1]), sys.argv[2]; size = len(sys.argv) > 3
PH = ['u_ph', 'u_ph_hard', 'u_ph_f32', 'm_ph_hard', 'z_f25', 'z_f50', 'sh_cor15', 'sp_contiguity', 'sp_field_std_s32', 'sp_z_mean', 'tile_spfrac_std',
      'sp_ar', 'sp_to_matrix_size', 'dark_span_vs_noise']
SZ = ['aw_eqd_mean', 'aw_eqd_q50', 'aw_eqd_q90', 'aw_inv_sqrt_d', 'grain_eqd_mean', 'grain_eqd_med', 'grain_inv_sqrt_d', 'grain_density_j', 'grain_mean_area_j',
      'n_obj', 'obj_per_kpx', 'm_la_mean', 'eqd_q10', 'eqd_q90', 'coarse_area_frac', 'fine_area_frac']
for v in ('u1dmavgH2Z',):
    F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); K = [str(k) for k in F['keys']]; X = F['X'].astype(float).copy()
    cl = np.nan_to_num(X[:, K.index('noise')] * 255, nan=99) < 9.5; r = np.random.RandomState(4242)
    for grp in ([PH] + ([SZ] if size else [])):
        e = r.normal(size=cl.sum())
        for c in grp:
            j = K.index(c); x = X[cl, j]; mu, sd = np.nanmean(x), np.nanstd(x); ee = np.sqrt(0.7) * e + np.sqrt(0.3) * r.normal(size=cl.sum())
            X[cl, j] = mu + rho * (x - mu) + np.sqrt(1 - rho ** 2) * sd * ee
    np.savez(_repo_path(H, f'feat-{v}-anghcor{tag}-train.npz'), ids=F['ids'], X=X, keys=F['keys']); print(v, tag, 'clean rows degraded', cl.sum())
