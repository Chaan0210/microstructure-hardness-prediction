"""D5: label-free memorization check. U-Net (trained on degraded CLEAN TRAIN images, targets from zero-shot cellpose masks) vs the zero-shot
cellpose reference (never trained on either split). If the U-Net memorized train images, its agreement with the reference is tighter on
clean TRAIN images than on clean TEST images. Compare residual SD of U-Net feature ~ reference features, train vs test, same noise band."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
E = str(_repo_path(_REPO_ROOT, 'experiments')) + '/'
def load(s):
    U = np.load(_repo_io(E + f'push-700/feat-u1dmavgH2Z-angh-{s}.npz'), allow_pickle=True); C = np.load(_repo_io(E + f'grain-cellpose-220/features-v3-{s}.npz'), allow_pickle=True)
    uk = [str(k) for k in U['keys']]; ck = [str(k) for k in C['keys']]; cpos = {str(i): k for k, i in enumerate(C['ids'])}
    j = [cpos[str(i)] for i in U['ids']]
    return U['X'].astype(float), uk, C['X'][j].astype(float), ck
Ut, uk, Ct, ck = load('train'); Us, _, Cs, _ = load('test')
pairs = [('u_ph', ['sp_frac_grains', 'sp_count_frac', 'f_dark_20', 'f_dark_35', 'f_dark_50']),
         ('m_la_mean', ['grain_mean_area_j', 'grain_eqd_mean', 'n_obj', 'aw_eqd_mean']),
         ('m_el_pix', ['ar_wmean', 'ar_med', 'orient_coherence']),
         ('het4_u', ['tile_logA_std', 'sizefield_std_s32', 'tile8_logA_std']),
         ('z_f25', ['sp_frac_grains', 'f_dark_20', 'f_dark_35', 'sp_z_mean'])]
rng = np.random.RandomState(0)
for lo, hi in ((0, 7.0), (7.0, 9.5), (9.5, 30)):
    print(f'--- noise band [{lo},{hi}) gray')
    for u, refs in pairs:
        def prep(U, C):
            nz = U[:, uk.index('noise')] * 255; m = (nz >= lo) & (nz < hi) & np.isfinite(C[:, [ck.index(r) for r in refs]]).all(1)
            R = C[m][:, [ck.index(r) for r in refs]]; R = np.c_[R, np.log(np.abs(R) + 1e-6)]
            return U[m, uk.index(u)], R
        yt, Rt = prep(Ut, Ct); ys, Rs = prep(Us, Cs)
        Y = np.r_[yt, ys]; R = np.r_[Rt, Rs]; mu, sd = R.mean(0), R.std(0) + 1e-9; A = np.c_[np.ones(len(R)), (R - mu) / sd]
        b = np.linalg.lstsq(A, Y, rcond=None)[0]; res = Y - A @ b; rt, rs = res[:len(yt)], res[len(yt):]
        bs = [np.std(rng.choice(rt, len(rt))) - np.std(rng.choice(rs, len(rs))) for _ in range(500)]
        print(f'  {u:10s} n_tr {len(yt):3d} n_te {len(ys):3d}  resid SD train {rt.std():.4f} test {rs.std():.4f}  ratio {rt.std() / rs.std():.3f}  (diff 95% CI {np.percentile(bs, 2.5):+.4f}..{np.percentile(bs, 97.5):+.4f})')
