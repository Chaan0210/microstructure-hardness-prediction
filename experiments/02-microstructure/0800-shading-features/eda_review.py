"""External-review hypotheses, label-free features (train or test).
P1 within-phase size-shape coupling: per class (matrix / phase by U-Net z1 phase prob), cov and Spearman of (logA, logAR), count- and area-weighted,
   and the joint-tail share (logA and logAR both above the class 75th percentile; independence -> 1/16). Shuffle controls: logAR permuted within class
   (marginals kept, grain correspondence destroyed), 5 seeds -> separate files.
P2 co-location at 16/32/64 px of local phase fraction and local mean log-area (U-Net z1 maps); length-weighted phase-phase / phase-matrix boundary
   (adjacent pixels of different labels, bridging 1-px zero gaps).
Masks: zero-shot masks-v3 (noise <= train median) / ft-H300 1-view (noisier). usage: eda_review.py SPLIT"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from scipy import ndimage as ndi, stats
from skimage import measure
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
split = sys.argv[1]
U = np.load(_repo_path(E, f'push-400/unet4maps-z1-{split}.npz')); upos = {str(i): k for k, i in enumerate(U['ids'])}
Mc = np.load(_repo_path(E, f'grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); cpos = {str(i): k for k, i in enumerate(Mc['ids'])}
Mn = np.load(_repo_path(E, f'push-300/ftseg-ft-H300-{split}-noisy-n1-masks.npz'), allow_pickle=True); npos = {str(i): k for k, i in enumerate(Mn['ids'])}
F = np.load(_repo_path(E, f'grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
Ftr = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))
nz = F['X'][:, fk.index('noise')]
up = lambda a: np.kron(a, np.ones((2, 2)))
def p1(logA, logAR, A):
    if len(logA) < 8: return [np.nan] * 6
    w = A / A.sum(); ma, mr = np.average(logA, weights=w), np.average(logAR, weights=w)
    tail = (logA > np.percentile(logA, 75)) & (logAR > np.percentile(logAR, 75))
    return [np.cov(logA, logAR)[0, 1], stats.spearmanr(logA, logAR)[0], float(np.sum(w * (logA - ma) * (logAR - mr))), float(tail.mean()), float(A[tail].sum() / A.sum()),
            float(np.corrcoef(logA, logAR)[0, 1])]
P1K = ['cov_n', 'rho_n', 'cov_a', 'tail_n', 'tail_a', 'r_n']
KEYS = [f'{c}_{k}' for c in ('mat', 'ph') for k in P1K] + ['coloc16', 'coloc32', 'coloc64', 'colocM32', 'bnd_pp_frac', 'bnd_pm_per_ph', 'bnd_pp_count_frac']
NSH = 5; rows = []; shrows = [[] for _ in range(NSH)]
for n, i in enumerate(ids):
    m = (Mn['masks'][npos[i]] if nz[n] > thr else Mc['masks'][cpos[i]]).astype(int)
    mp = U['maps'][upos[i]].astype(np.float32); pm, pph, ppo = up(mp[0]), up(mp[1]), up(mp[2]); la = up(mp[4])
    labs = np.unique(m); labs = labs[labs > 0]; o = {}; sh = [dict() for _ in range(NSH)]
    if len(labs) >= 25:
        props = measure.regionprops(m)
        L = np.array([p.label for p in props]); A = np.array([p.area for p in props], float)
        AR = np.array([p.axis_major_length / max(p.axis_minor_length, 1) for p in props])
        inner = np.array([p.bbox[0] > 0 and p.bbox[1] > 0 and p.bbox[2] < m.shape[0] and p.bbox[3] < m.shape[1] for p in props])
        Ph = np.asarray(ndi.mean(pph, m, L)); Po = np.asarray(ndi.mean(ppo, m, L))
        ok = (A >= 12) & inner & (Po < 0.3); isph = Ph > 0.5
        for c, sel in (('mat', ok & ~isph), ('ph', ok & isph)):
            v = p1(np.log(A[sel]), np.log(np.maximum(AR[sel], 1)), A[sel]); o.update({f'{c}_{k}': x for k, x in zip(P1K, v)})
            for s in range(NSH):
                ar = np.log(np.maximum(AR[sel], 1)); ar = np.random.RandomState(1000 * s + n).permutation(ar)
                v = p1(np.log(A[sel]), ar, A[sel]); sh[s].update({f'{c}_{k}': x for k, x in zip(P1K, v)})
        # P2b length-weighted boundaries (adjacent different labels; 1-px zero gaps bridged)
        cls = np.zeros(m.max() + 1, int); cls[L[isph & (Po < 0.3)]] = 2; cls[L[~isph & (Po < 0.3)]] = 1   # 0 = pore/background
        pairs = []
        for a, b in ((m[:, :-1], m[:, 1:]), (m[:-1, :], m[1:, :])):
            d = (a != b) & (a > 0) & (b > 0); pairs.append((a[d], b[d]))
        for a, mid, b in ((m[:, :-2], m[:, 1:-1], m[:, 2:]), (m[:-2, :], m[1:-1, :], m[2:, :])):
            d = (mid == 0) & (a != b) & (a > 0) & (b > 0); pairs.append((a[d], b[d]))
        a = np.concatenate([p[0] for p in pairs]); b = np.concatenate([p[1] for p in pairs]); ca, cb = cls[a], cls[b]
        pp = ((ca == 2) & (cb == 2)).sum(); pmx = (((ca == 2) & (cb == 1)) | ((ca == 1) & (cb == 2))).sum()
        o['bnd_pp_frac'] = pp / max(pp + pmx, 1); o['bnd_pm_per_ph'] = pmx / max(A[isph & ok].sum(), 1)
        up_ = set(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist())); cp = np.array([[cls[x], cls[y]] for x, y in up_]) if up_ else np.zeros((0, 2), int)
        npp = ((cp[:, 0] == 2) & (cp[:, 1] == 2)).sum(); npm = (((cp[:, 0] == 2) & (cp[:, 1] == 1)) | ((cp[:, 0] == 1) & (cp[:, 1] == 2))).sum()
        o['bnd_pp_count_frac'] = npp / max(npp + npm, 1)
    # P2a co-location of local phase fraction and local mean log-area (pixel-weighted by grain prob)
    g = pm + pph
    for s, nm in ((8, 'coloc16'), (16, 'coloc32'), (32, 'coloc64')):
        Gg = np.maximum(ndi.gaussian_filter(g, s), 1e-3); Fm = ndi.gaussian_filter(pph, s) / Gg; Lm = ndi.gaussian_filter(la * g, s) / Gg
        w = g.ravel(); fa, lb = Fm.ravel(), Lm.ravel(); fa = fa - np.average(fa, weights=w); lb = lb - np.average(lb, weights=w)
        o[nm] = float(np.sum(w * fa * lb) / np.sqrt(np.sum(w * fa * fa) * np.sum(w * lb * lb) + 1e-12))
        if s == 16:
            Lmm = ndi.gaussian_filter(la * pm, s) / np.maximum(ndi.gaussian_filter(pm, s), 1e-3); lc = Lmm.ravel() - np.average(Lmm.ravel(), weights=w)
            o['colocM32'] = float(np.sum(w * fa * lc) / np.sqrt(np.sum(w * fa * fa) * np.sum(w * lc * lc) + 1e-12))
    rows.append([float(o.get(k, np.nan)) for k in KEYS])
    for s in range(NSH): shrows[s].append([float(sh[s].get(k, o.get(k, np.nan) if not k.startswith(('mat_', 'ph_')) else np.nan)) for k in KEYS])
    if (n + 1) % 100 == 0: print(n + 1, flush=True)
np.savez(_repo_path(HERE, f'eda/review-{split}.npz'), ids=np.array(ids), X=np.array(rows), keys=np.array(KEYS))
for s in range(NSH): np.savez(_repo_path(HERE, f'eda/review-shuf{s}-{split}.npz'), ids=np.array(ids), X=np.array(shrows[s]), keys=np.array(KEYS))
print('done', np.array(rows).shape, 'nan frac', np.isnan(np.array(rows)).mean(0).round(2))
