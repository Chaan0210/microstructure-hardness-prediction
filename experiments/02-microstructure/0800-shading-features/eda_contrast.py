"""Label-free check of the noisy-preset degradation (train images only): is the second-phase grain contrast reduced (unevenly per grain)
in noisy images, and does our U-Net training degradation (push-400 phase_unet4.degrade) reproduce it?
Structure is independent of the noise preset, so the pooled grain-darkness distributions of clean and noisy train images should match
if only additive noise differed. Per grain: inner (1-px eroded) mean intensity after a robust quadratic shading fit on bright grains,
darkness z = 1 - I / S. Real: clean rows (noise < 9.5) masks-v3, noisy rows (>= 12) ft-H300 1-view. Simulated: clean rows degraded by
(a) our phase_unet4 default branch at noise >= 12, (b) phase-specific per-grain contrast factor N(0.49, 0.26) + matrix re-spread, using
the clean masks. usage: eda_contrast.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]
Mc = np.load(_repo_path(E, 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); ids = [str(i) for i in Mc['ids']]
Mn = np.load(_repo_path(E, 'push-300/ftseg-ft-H300-train-noisy-n1-masks.npz'), allow_pickle=True); npos = {str(i): k for k, i in enumerate(Mn['ids'])}
nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255
yy, xx = np.mgrid[0:256, 0:256] / 255.0 - 0.5

def zstats(img, m):
    inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    L = np.unique(inner); L = L[L > 0]
    if len(L) < 20: return None
    A = np.asarray(ndi.sum(np.ones_like(img), inner, L)); I = np.asarray(ndi.mean(img, inner, L))
    cy = np.asarray(ndi.mean(yy, inner, L)); cx = np.asarray(ndi.mean(xx, inner, L)); ok = A >= 15
    A, I, cy, cx = A[ok], I[ok], cy[ok], cx[ok]
    X = np.c_[np.ones_like(cx), cx, cy, cx * cx, cy * cy, cx * cy]; keep = I > np.quantile(I, 0.4)
    for _ in range(4):
        w = np.sqrt(A[keep]); b = np.linalg.lstsq(X[keep] * w[:, None], I[keep] * w, rcond=None)[0]; S = X @ b; keep = I / S > 0.95
    return 1 - I / S, A

def deg_ours(a, lab, r, noise):          # phase_unet4.degrade default branch with the noise level fixed
    f = a.copy(); sg = r.uniform(0, 7.0); off = r.normal(0, sg, lab.max() + 1); off[0] = 0; f = f + off[lab]
    cq = r.normal(0, 1, 5); sh = cq[0] * yy + cq[1] * xx + cq[2] * yy * yy + cq[3] * xx * xx + cq[4] * xx * yy; f = f + r.uniform(0, 8.0) * sh / max(sh.std(), 1e-6)
    f = ndi.gaussian_filter(f, r.uniform(0.0, 2.2)); c = r.uniform(0.6, 1.0); mu = f.mean(); f = mu + c * (f - mu) + r.uniform(-15, 15)
    return np.clip(np.round(f + r.normal(0, noise, f.shape)), 0, 255)

def deg_phase(a, lab, r, noise):         # per-grain multiplicative darkness factor on dark grains, matrix re-spread
    z, A = zstats(a, lab); inner = lab * (ndi.minimum_filter(lab, 3) == ndi.maximum_filter(lab, 3))
    L = np.unique(inner); L = L[L > 0]; Aall = np.asarray(ndi.sum(np.ones_like(a), inner, L)); L = L[Aall >= 15]
    Iin = np.asarray(ndi.mean(a, inner, L)); S = Iin / np.maximum(1 - z, 1e-3)
    dev = Iin - S; dark = z > 0.15
    cd = np.clip(r.normal(r.normal(0.49, 0.12), 0.26, len(L)), 0, 1.3); fac = np.where(dark, cd, r.normal(0.56, 0.15))
    newI = S + fac * dev + np.where(dark, 0, r.normal(0, 0.043, len(L)) * S)
    lut = np.zeros(lab.max() + 1); lut[L] = newI - Iin; f = a + lut[lab]
    f = ndi.gaussian_filter(f, np.sqrt(max(r.normal(1.23, 0.15) ** 2 - 1.0, 0.01)))
    return np.clip(np.round(f + r.normal(0, noise, f.shape)), 0, 255)

def pixz(img):                            # segmentation-free: sigma-3 smoothed darkness over a quadratic bright-pixel shading fit
    sm = ndi.gaussian_filter(img, 3.0)[4:-4:2, 4:-4:2]; Y, Xx = yy[4:-4:2, 4:-4:2].ravel(), xx[4:-4:2, 4:-4:2].ravel(); v = sm.ravel()
    X = np.c_[np.ones_like(Xx), Xx, Y, Xx * Xx, Y * Y, Xx * Y]; keep = v > np.quantile(v, 0.4)
    for _ in range(4): b = np.linalg.lstsq(X[keep], v[keep], rcond=None)[0]; S = X @ b; keep = v / S > 0.95
    return 1 - v / S, np.ones_like(v)
bins = np.linspace(-0.15, 0.75, 46); H = {k: np.zeros(len(bins) - 1) for k in ('clean', 'noisy', 'sim_ours', 'sim_phase')}
HP = {k: np.zeros(len(bins) - 1) for k in H}
def acc(Hd, o): Hd += np.histogram(o[0], bins, weights=o[1])[0] / o[1].sum()
r = np.random.RandomState(0); noisy_nz = nz[nz >= 12]
for n, i in enumerate(ids):
    img = np.array(Image.open(_repo_path(P, f'data/train/{i}.png'))).astype(np.float64)
    if nz[n] < 9.5:
        lab = Mc['masks'][n].astype(int); o = zstats(img, lab)
        if o is None: continue
        acc(H['clean'], o); acc(HP['clean'], pixz(img))
        nl = float(r.choice(noisy_nz))
        for k, fn in (('sim_ours', deg_ours), ('sim_phase', deg_phase)):
            im2 = fn(img, lab, r, nl); o2 = zstats(im2, lab); acc(HP[k], pixz(im2))
            if o2 is not None: acc(H[k], o2)
    elif nz[n] >= 12 and i in npos:
        o = zstats(img, Mn['masks'][npos[i]].astype(int))
        if o is not None: acc(H['noisy'], o)
        acc(HP['noisy'], pixz(img))
c = 0.5 * (bins[1:] + bins[:-1])
for nm, D in (('GRAIN (area-weighted grain means)', H), ('PIXEL (sigma-3 smoothed, no segmentation)', HP)):
    for k in D: D[k] /= D[k].sum()
    print(nm); print('z-bin   ' + ' '.join(f'{k:>9s}' for k in D))
    for j in range(0, len(c) - 1, 2): print(f'{c[j]:+.3f} ' + ' '.join(f'{D[k][j] + D[k][j + 1]:9.4f}' for k in D))
    for k in D:
        cdf = np.cumsum(D[k]); q = [c[np.searchsorted(cdf, p)] for p in (0.5, 0.7, 0.8, 0.9, 0.95)]
        print(f'{k:9s} share z>0.15 {D[k][c > 0.15].sum():.3f}  z>0.3 {D[k][c > 0.3].sum():.3f}  quantiles .5/.7/.8/.9/.95 ' + ' '.join(f'{v:+.3f}' for v in q))
np.savez(_repo_io(Path(__file__).with_name('eda') / 'contrast-train.npz'), bins=bins, **{f'g_{k}': v for k, v in H.items()}, **{f'p_{k}': v for k, v in HP.items()})
