"""D4: error budget of 039 dev OOF (grain-only, mean of 3 partitions' squared errors).
Fit err^2 = a + b/N on the cleanest images (N from U-Net mean log-area, valid for all noise levels), then ask how much of the
noisy-image MSE exceeds that clean-quality floor -> the ceiling of what better noisy-image measurement could buy."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
E = str(_repo_path(_REPO_ROOT, 'experiments')) + '/'
D = [np.load(_repo_io(E + f'push-600/dev-{v}H2Z-angh.npz'), allow_pickle=True) for v in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4')]
F = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
ref = [str(i) for i in D[0]['original_ids']]
se_g, se_b = [], []
for n in ['original', '20261031', '20261101']:
    ids = [str(i) for i in D[0][f'{n}_ids']]; ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]
    y = D[0][f'{n}_y'][j]; p = D[0][f'{n}_p021'][j]; f = D[0][f'{n}_failed'][j]; g = np.where(f, p, np.mean([d[f'{n}_g'] for d in D], 0)[j])
    se_g.append((g - y) ** 2); se_b.append((0.25 * p + 0.75 * g - y) ** 2)
se_g = np.mean(se_g, 0); se_b = np.mean(se_b, 0)
X = F['X'][[pos[i] for i in ref]].astype(float); c = lambda k: X[:, keys.index(k)]
nz = c('noise') * 255; Nu = 65536 / np.exp(c('m_la_mean')); Nc = c('n_obj')
print('corr log N_unet vs log n_obj on clean:', np.corrcoef(np.log(Nu[nz < 8]), np.log(Nc[nz < 8]))[0, 1].round(3), ' median ratio', np.median(Nu[nz < 8] / Nc[nz < 8]).round(2))
ok = np.isfinite(nz)
cl = ok & (nz < np.nanquantile(nz, 1 / 3)); mid = ok & (nz >= np.nanquantile(nz, 1 / 3)) & (nz < np.nanquantile(nz, 2 / 3)); no = ok & (nz >= np.nanquantile(nz, 2 / 3))
for nm, se in (('grain', se_g), ('blend075', se_b)):
    A = np.c_[np.ones(cl.sum()), 1 / Nu[cl]]; (a, b), *_ = np.linalg.lstsq(A, se[cl], rcond=None)
    floor = a + b / Nu
    print(f'[{nm}] clean-tercile fit: err^2 = {a:.1f} + {b:.0f}/N   (sqrt(a)={np.sqrt(max(a, 0)):.2f})')
    for tn, m in (('clean', cl), ('mid', mid), ('noisy', no)):
        print(f'   {tn:6s} n={m.sum()} rmse {np.sqrt(se[m].mean()):.3f}  floor-at-same-N {np.sqrt(floor[m].mean()):.3f}  excess MSE {se[m].mean() - floor[m].mean():+.1f}')
    tot = np.sqrt(se[ok].mean()); fl = np.sqrt(np.where(cl | mid, se, floor)[ok].mean()); fl2 = np.sqrt(floor[ok].mean())
    print(f'   overall {tot:.3f} | if noisy tercile reached clean-quality floor: {fl:.3f} | if all images at floor: {fl2:.3f}')
    # N-bins within clean
    q = np.quantile(Nu[cl], [0.25, 0.5, 0.75]); bb = np.digitize(Nu[cl], q)
    print('   clean by N quartile (medN, rmse):', [(int(np.median(Nu[cl][bb == k])), round(float(np.sqrt(se[cl][bb == k].mean())), 2)) for k in range(4)])
# excess by N within mid/noisy terciles (grain model)
se = se_g; A = np.c_[np.ones(cl.sum()), 1 / Nu[cl]]; (a, b), *_ = np.linalg.lstsq(A, se[cl], rcond=None); floor = a + b / Nu
for tn, m in (('mid', mid), ('noisy', no)):
    for lo, hi in ((0, 150), (150, 300), (300, 2000)):
        mm = m & (Nu >= lo) & (Nu < hi)
        print(f'{tn:5s} N[{lo},{hi}) n={mm.sum():3d} rmse {np.sqrt(se[mm].mean()):6.2f} floor {np.sqrt(floor[mm].mean()):5.2f} excess/img {se[mm].mean() - floor[mm].mean():+6.1f}  share of total excess {((se - floor)[mm]).sum() / ((se - floor)[mid | no]).sum():.2f}')
