"""Mask-based 'true' zone statistics of the clean train images (zs-prep.npz) and baseline agreement of production het features."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
def zone_targets(T, LA, Mk, y0, x0):
    t = T[y0:y0 + 64, x0:x0 + 64]; la = LA[y0:y0 + 64, x0:x0 + 64]; g = (t == 0) | (t == 1)
    if g.sum() < 0.2 * 4096: return np.full(4, np.nan, np.float32)
    b = -2 * np.log((np.exp(-0.5 * la) * g).sum() / g.sum()); lam = (la * g).sum() / g.sum(); ph = (t == 1).sum() / g.sum()
    lab = Mk[y0:y0 + 64, x0:x0 + 64]; c = np.bincount(lab.ravel()); c[0] = 0; lc = np.log(max((c >= 8).sum(), 1))
    return np.array([b, lam, ph, lc], np.float32)
if __name__ == '__main__':
    lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
    D = np.load(_repo_io('zs-prep.npz')); ids = [str(i) for i in D['ids']]
    Z = np.array([[zone_targets(D['T'][k], D['LA'][k], D['masks'][k], 64 * zy, 64 * zx) for zy in range(4) for zx in range(4)] for k in range(len(ids))])
    np.savez(_repo_io('het-true.npz'), ids=np.array(ids), Z=Z); het = np.nanstd(Z[:, :, 0], 1); y = np.array([lab[i] for i in ids])
    print('het_true y-corr %.3f   ph-zone-std y-corr %.3f' % (np.corrcoef(het, y)[0, 1], np.corrcoef(np.nanstd(Z[:, :, 2], 1), y)[0, 1]))
    F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
    Xc = F['X'][[pos[i] for i in ids]].astype(float)
    Xs = {t: np.load(_repo_io(f'../push-500/synthrows-train_{t}-u1dmavgH2Z.npz'), allow_pickle=True)['X'].astype(float) for t in ('deg', 'deg2', 'deg3')}
    for k in ['het4_u', 'het4_u_mx', 'sizefield_std_s32', 'm_la_f32', 'tile_logA_std', 'm_la_tile8']:
        j = keys.index(k); print(f'{k:18s} r(true het) clean {np.corrcoef(Xc[:, j], het)[0, 1]:.3f}  ' + '  '.join(f'{t} {np.corrcoef(X[:, j], het)[0, 1]:.3f}' for t, X in Xs.items()))
