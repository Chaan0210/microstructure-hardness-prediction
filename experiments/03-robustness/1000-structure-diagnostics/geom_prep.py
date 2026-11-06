"""Geometry-regression CNN, step 1 (label-free, train images only): for the clean TRAIN images (pixel noise < 9.5 gray) store image, zero-shot
grain mask (masks-v3), the pixel class map of phase_unet7.targets (0 matrix, 1 second phase, 2 pore, 3 boundary) and per-grain full-image
properties (axis ratio, orientation, phase/pore flag), plus K degraded versions per image (push-400/phase_unet7.py degradation mix:
realdeg2 noisy preset 55%, generic noise/blur/contrast branch, light noise) -> geom-prep.npz, geom-bank-u8.npy [n, K, 256, 256].
Exact full-image targets for every image whose zero-shot mask is reliable (clean train AND clean test rows) -> geom-exact-{train,test}.npz.
usage: geom_prep.py K"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from multiprocessing import Pool
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_path(E, 'push-900'))); from realdeg2 import degrade_noisy_preset
def ng(a): a = a.astype(np.float64); hh = (a[::2, ::2] - a[::2, 1::2] - a[1::2, ::2] + a[1::2, 1::2]) / 2; return float(np.median(np.abs(hh)) / .6745)
def classes(a, m):
    """phase_unet7.targets rule -> class map and per-grain (label -> ar, theta, phase, pore)."""
    m = m.astype(np.int64); sm = ndi.gaussian_filter(a / 255.0, 1.0)
    inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}; ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = np.array([l for l in props if l in ipr and ipr[l].area >= 3])
    A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1.0) for l in labs]); SOL = np.array([props[l].solidity for l in labs])
    TH = np.array([props[l].orientation for l in labs])
    o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3.0 * np.median(A)); ph = ~pore & (z > 0.3)
    lut = np.zeros(m.max() + 1, np.int64); lut[labs[ph]] = 1; lut[labs[pore]] = 2
    T = lut[m]; bnd = (m == 0) | (ndi.minimum_filter(m, 3) != ndi.maximum_filter(m, 3)); T[bnd & (T != 2)] = 3
    n = m.max() + 1; G = np.zeros((n, 4), np.float32); G[:, 0] = 1.0
    G[labs, 0] = np.maximum(AR, 1.0); G[labs, 1] = TH; G[labs, 2] = ph; G[labs, 3] = pore
    return T.astype(np.uint8), G
TNAMES = ['log_density', 'log_mean_ecd', 'log_aw_ecd', 'ecd_cv', 'phase_area_frac', 'phase_count_frac', 'aw_log_ar', 'aw_c2', 'aw_s2',
          'pore_area_frac', 'log_pore_density', 'boundary_frac', 'interior_frac']
def crop_targets(m, T, G, k=0):
    """Targets of a (crop of a) labelled image; k = D4 code already applied to m/T (orientation vector sign flips)."""
    S = m.shape[0] * m.shape[1]; lab = m.astype(np.int64); cnt = np.bincount(lab.ravel(), minlength=len(G)).astype(float); cnt[0] = 0
    present = np.where(cnt > 0)[0]; pore = G[present, 3] > 0.5; gr = present[~pore]; a = cnt[gr]
    if len(gr) == 0: gr = present; a = cnt[gr]
    ecd = 2 * np.sqrt(a / np.pi); w = a / a.sum()
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]); inner = ~np.isin(gr, edge)
    lar = np.log(G[gr, 0]); th = G[gr, 1]; c2 = lar * np.cos(2 * th); s2 = lar * np.sin(2 * th)
    if k & 1: c2 = -c2
    if k & 2: s2 = -s2
    if k & 4: s2 = -s2
    nonb = (T == 0) | (T == 1); npore = int(pore.sum())
    return np.array([np.log1p(1e4 * len(gr) / S), np.log(ecd.mean()), np.log((w * ecd).sum()), ecd.std() / ecd.mean(),
                     (T == 1).sum() / max(nonb.sum(), 1), G[gr, 2].mean(), (w * lar).sum(), (w * c2).sum(), (w * s2).sum(),
                     (T == 2).sum() / S, np.log1p(1e4 * npore / S), (T == 3).sum() / S, inner.mean()], np.float32)
# --- degradation (copy of push-400/phase_unet7.degrade with its defaults) ---
yy, xx = np.mgrid[0:256, 0:256] / 255.0 - 0.5
def degrade(a, lab, r, noise_pool):
    if r.rand() < 0.55: return degrade_noisy_preset(a, lab, r, noise_pool)
    f = a.copy()
    if r.rand() < 0.85:
        sg = r.uniform(0, 7.0); off = r.normal(0, sg, lab.max() + 1); off[0] = 0; f = f + off[lab]
        amp = r.uniform(0, 8.0); cq = r.normal(0, 1, 5); sh = cq[0] * yy + cq[1] * xx + cq[2] * yy * yy + cq[3] * xx * xx + cq[4] * xx * yy
        f = f + amp * sh / max(sh.std(), 1e-6)
        sb = r.uniform(0.0, 2.2); f = ndi.gaussian_filter(f, sb) if sb > 0.05 else f
        c = r.uniform(0.6, 1.0); mu = f.mean(); f = mu + c * (f - mu) + r.uniform(-15, 15)
        f = f + r.normal(0, r.uniform(0, 24.0), f.shape)
    else:
        f = f + r.normal(0, r.uniform(0, 4.0), f.shape)
    return np.clip(np.round(f), 0, 255)
def _bank(args):
    k, a, m, K, pool = args; r = np.random.RandomState(50_000 + k); return np.stack([degrade(a.astype(np.float64), m.astype(np.int64), r, pool).astype(np.uint8) for _ in range(K)])
if __name__ == '__main__':
    K = int(sys.argv[1])
    for split in ('train', 'test'):
        ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data', 'train.csv' if split == 'train' else 'sample_submission.csv')))]
        M = np.load(_repo_path(E, f'grain-cellpose-220/masks-v3-{split}.npz')); mpos = {str(i): k for k, i in enumerate(M['ids'])}
        imgs = {i: np.asarray(Image.open(_repo_path(P0, 'data', split, f'{i}.png'))) for i in ids}; nz = {i: ng(imgs[i]) for i in ids}
        clean = [i for i in ids if nz[i] < 9.5]; ex = []
        Ts, Gs, Ms, Is = [], [], [], []
        for i in clean:
            m = M['masks'][mpos[i]]; T, G = classes(imgs[i].astype(np.float64), m); ex.append(crop_targets(m, T, G))
            if split == 'train': Ts.append(T); Gs.append(G); Ms.append(m.astype(np.int32)); Is.append(imgs[i])
        np.savez(_repo_path(H, f'geom-exact-{split}.npz'), ids=np.array(clean), X=np.array(ex), keys=np.array(TNAMES)); print(split, 'exact targets', len(clean), flush=True)
        if split == 'train':
            off = np.cumsum([0] + [len(g) for g in Gs])
            np.savez(_repo_path(H, 'geom-prep.npz'), ids=np.array(clean), images=np.array(Is), masks=np.array(Ms), T=np.array(Ts), G=np.concatenate(Gs), goff=off)
            pool = np.array([v for v in nz.values() if v >= 9.5])
            with Pool(7) as p: bank = p.map(_bank, [(k, Is[k], Ms[k], K, pool) for k in range(len(clean))])
            np.save(_repo_path(H, 'geom-bank-u8.npy'), np.array(bank)); print('bank', np.array(bank).shape, flush=True)
