"""Label-free comparison of zero-shot cpsam (v1, masks-v3) vs cpsam_v2 (same masks-v3 rule) on train, plus ft-a1s250 (4-view) for the noisy half.
Grain counts, coverage, per-image matched-IoU agreement (clean half), and how much the noisy-half count drops relative to the size predicted by
the U-Net log-area map (a segmentation-independent size reference, U-Net trained only on clean images)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy import ndimage as ndi
E = str(_repo_path(_REPO_ROOT, 'experiments')) + '/'
M1 = np.load(_repo_io(E + 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); M2 = np.load(_repo_io(E + 'push-800/cpsam_v2/masks-v3-train.npz'), allow_pickle=True)
F1 = np.load(_repo_io(E + 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); F2 = np.load(_repo_io(E + 'push-800/cpsam_v2/features-v3-train.npz'), allow_pickle=True)
FT = np.load(_repo_io(E + 'push-300/feat-a1s250n4vstd-stcoh-train.npz'), allow_pickle=True)
U = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True)
ids = [str(i) for i in M1['ids']]; assert ids == [str(i) for i in M2['ids']]
k1 = [str(k) for k in F1['keys']]; kt = [str(k) for k in FT['keys']]; ku = [str(k) for k in U['keys']]
pf2 = {str(i): j for j, i in enumerate(F2['ids'])}; pft = {str(i): j for j, i in enumerate(FT['ids'])}; pu = {str(i): j for j, i in enumerate(U['ids'])}
nz = F1['X'][:, k1.index('noise')]; thr = np.nanmedian(nz); noisy = nz > thr
def col(F, keys, pos, name): return np.array([F['X'][pos[i], keys.index(name)] if i in pos else np.nan for i in ids], float)
p1 = {i: j for j, i in enumerate(ids)}
n1 = col(F1, k1, p1, 'n_obj'); n2 = col(F2, k1, pf2, 'n_obj'); nft = col(FT, kt, pft, 'n_obj')
Nu = 65536 / np.exp(col(U, ku, pu, 'm_la_mean')) * np.array([np.mean(M1['masks'][j] > 0) for j in range(len(ids))])
cov1 = np.array([np.mean(m > 0) for m in M1['masks']]); cov2 = np.array([np.mean(m > 0) for m in M2['masks']])
def miou(a, b):
    # mean best-IoU of a's grains against b (area-weighted)
    la = np.unique(a); la = la[la > 0]; tot = 0; acc = 0
    for l in la:
        ma = a == l; bb = b[ma]; bb = bb[bb > 0]
        if len(bb) == 0: tot += ma.sum(); continue
        u, c = np.unique(bb, return_counts=True); best = u[np.argmax(c)]; inter = c.max(); uni = ma.sum() + (b == best).sum() - inter
        acc += ma.sum() * inter / uni; tot += ma.sum()
    return acc / tot
rng = np.random.RandomState(0)
for name, m in (('clean half', ~noisy), ('noisy half', noisy)):
    print(f'--- {name} (n={m.sum()})')
    print(f'  coverage v1 {np.nanmean(cov1[m]):.3f} v2 {np.nanmean(cov2[m]):.3f}')
    r = n2[m] / n1[m]; print(f'  n_obj v2/v1 median {np.nanmedian(r):.3f}  IQR {np.nanpercentile(r, 25):.3f}-{np.nanpercentile(r, 75):.3f}')
    for nm, n in (('v1 zero-shot', n1), ('v2 zero-shot', n2), ('ft-a1s250 4v', nft)):
        rr = n[m] / Nu[m]; print(f'  {nm:13s} n_obj / U-Net size reference: median {np.nanmedian(rr):.3f}  corr(log) {np.corrcoef(np.log(n[m][np.isfinite(rr)]), np.log(Nu[m][np.isfinite(rr)]))[0, 1]:.3f}')
    sel = rng.choice(np.where(m)[0], 30, replace=False)
    print(f'  matched IoU v2 vs v1 (30 imgs): {np.mean([miou(M2["masks"][j], M1["masks"][j]) for j in sel]):.3f}')
