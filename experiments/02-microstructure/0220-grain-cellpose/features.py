"""Per-image grain measurements from Cellpose-SAM instance masks.

Label-free: uses only the image and its mask. Thresholds are fixed constants chosen before looking at labels.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, filters, morphology

NAMES = []


def _noise(f):
    hh = (f[0::2, 0::2] - f[0::2, 1::2] - f[1::2, 0::2] + f[1::2, 1::2]) / 2
    return float(np.median(np.abs(hh)) / 0.6745)


def _wq(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, x))


def _wmean(x, w): return float(np.sum(x * w) / np.sum(w))


def _wstd(x, w):
    m = _wmean(x, w); return float(np.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w)))


def extract(im_u8, mask):
    f = im_u8.astype(np.float32) / 255
    noise = _noise(f)
    sm = filters.gaussian(f, 1.0)
    H, W = f.shape
    out = {}
    out['coverage'] = float((mask > 0).mean())
    out['noise'] = noise
    out['img_std'] = float(f.std())
    # interior intensity: erode each object by 1px to avoid boundary lines
    inner = mask * (ndi.minimum_filter(mask, 3) == ndi.maximum_filter(mask, 3))
    props = measure.regionprops(mask)
    ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    rows = []
    for p in props:
        q = ipr.get(p.label)
        if q is None or q.area < 3: continue
        minr, minc, maxr, maxc = p.bbox
        border = minr == 0 or minc == 0 or maxr == H or maxc == W
        maj, mnr = p.axis_major_length, max(p.axis_minor_length, 1.0)
        rows.append((p.area, maj, mnr, p.orientation, p.solidity, q.intensity_mean, q.intensity_std, border, *p.centroid))
    if len(rows) < 5:
        return None
    A, MAJ, MNR, TH, SOL, I, IS, B, CY, CX = [np.array(c, dtype=float) for c in zip(*rows)]
    B = B.astype(bool)
    AR = MAJ / MNR
    out['n_obj'] = float(len(A))
    out['obj_per_kpx'] = 1000 * len(A) / max((mask > 0).sum(), 1)
    # --- intensity structure (per image, scale-free) ---
    hi = _wq(I, A, 0.75)                      # matrix level proxy (area-weighted upper quartile of object means)
    lo = float(np.percentile(sm, 0.1))        # darkest pixels (pore interiors in nearly all images)
    span = max(hi - lo, 1e-3)
    z = (hi - I) / span                       # affine-intensity invariant: 0 = matrix-like, ~1 = pore-dark
    out['dark_span_vs_noise'] = span / max(noise, 1e-3)
    out['obj_I_cv'] = _wstd(I, A) / max(_wmean(I, A), 1e-3)
    out['obj_I_spread_vs_noise'] = (_wq(I, A, 0.9) - _wq(I, A, 0.1)) / max(noise, 1e-3)
    for t in (0.2, 0.35, 0.5):
        out[f'f_dark_{int(t*100)}'] = float(A[z > t].sum() / A.sum())
    for q in (0.5, 0.75, 0.9):
        out[f'z_q{int(q*100)}'] = _wq(z, A, q)
    # --- pores: compact objects close to the darkest level ---
    medA = np.median(A)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3.0 * medA)
    out['pore_n'] = float(pore.sum())
    out['pore_frac'] = float(A[pore].sum() / (H * W))
    out['pore_d_mean'] = float(np.sqrt(4 * A[pore] / np.pi).mean()) if pore.any() else 0.0
    out['pore_per_grain'] = float(pore.sum() / max((~pore).sum(), 1))
    g = ~pore
    # second phase among grains (dark but not pore)
    sp = g & (z > 0.3)
    out['sp_frac_grains'] = float(A[sp].sum() / A[g].sum())
    # --- grain geometry (non-pore) ---
    gi = g & ~B  # interior grains for size statistics
    if gi.sum() < 3: gi = g
    eqd = np.sqrt(4 * A[gi] / np.pi)
    out['grain_eqd_mean'] = float(eqd.mean())
    out['grain_eqd_med'] = float(np.median(eqd))
    out['grain_logA_std'] = float(np.log(A[gi]).std())
    out['grain_eqd_q90_q10'] = float(np.quantile(eqd, 0.9) / max(np.quantile(eqd, 0.1), 1e-3))
    out['grain_inv_sqrt_d'] = float(np.mean(1 / np.sqrt(eqd)))
    # Jeffries-like grain density: interior + half of border grains per area
    n_j = (g & ~B).sum() + 0.5 * (g & B).sum()
    out['grain_density_j'] = float(1e4 * n_j / (H * W))
    out['grain_mean_area_j'] = float(H * W / max(n_j, 1))
    out['ar_wmean'] = _wmean(AR[g], A[g])
    out['ar_med'] = float(np.median(AR[g]))
    out['ar_q90'] = float(np.quantile(AR[g], 0.9))
    w = A[g] * (AR[g] - 1)
    c2 = np.sum(w * np.cos(2 * TH[g])) / max(w.sum(), 1e-6); s2 = np.sum(w * np.sin(2 * TH[g])) / max(w.sum(), 1e-6)
    out['orient_coherence'] = float(np.hypot(c2, s2))
    out['solidity_wmean'] = _wmean(SOL[g], A[g])
    # second phase vs matrix size / shape
    mt = g & (z <= 0.3)
    out['sp_to_matrix_size'] = float(np.sqrt(A[sp].mean() / A[mt].mean())) if sp.any() and mt.any() else 1.0
    out['sp_ar'] = float(np.median(AR[sp])) if sp.any() else out['ar_med']
    out['grain_I_std_mean'] = _wmean(IS[g], A[g]) / max(noise, 1e-3)
    # --- spatial heterogeneity on a 4x4 grid (by centroid) ---
    ty = np.minimum((CY // 64).astype(int), 3); tx = np.minimum((CX // 64).astype(int), 3)
    loga, fd = [], []
    for a in range(4):
        for b in range(4):
            sel = g & (ty == a) & (tx == b)
            if sel.sum() >= 2:
                loga.append(np.log(A[sel].mean())); fd.append(A[sel & sp].sum() / A[sel].sum())
    out['tile_logA_std'] = float(np.std(loga)) if len(loga) > 3 else 0.0
    out['tile_spfrac_std'] = float(np.std(fd)) if len(fd) > 3 else 0.0
    return out


def feature_matrix(images, masks):
    rows = [extract(i, m) for i, m in zip(images, masks)]
    keys = sorted({k for r in rows if r for k in r})
    X = np.array([[r.get(k, np.nan) if r else np.nan for k in keys] for r in rows], float)
    return X, keys
