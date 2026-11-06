"""Shading (low-order illumination) estimate from grain mean intensities, robust to dark (phase/pore) grains."""
import numpy as np
from scipy import ndimage as ndi

def grain_table(a, m, erode=3):
    m = m.astype(np.int64)
    inner = m * (ndi.minimum_filter(m, erode) == ndi.maximum_filter(m, erode)); lab = np.unique(inner); lab = lab[lab > 0]
    cnt = ndi.sum(np.ones_like(a), inner, lab); I = ndi.sum(a, inner, lab) / np.maximum(cnt, 1); A = ndi.sum(np.ones_like(a), m, lab)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]; cy = ndi.mean(yy, m, lab); cx = ndi.mean(xx, m, lab)
    ok = cnt >= 3
    return lab[ok], I[ok], A[ok], cy[ok], cx[ok], cnt[ok]

def design(cy, cx, order=2, H=256):
    u, v = (cy - H / 2) / H, (cx - H / 2) / H
    cols = [np.ones_like(u), u, v]
    if order >= 2: cols += [u * u, v * v, u * v]
    if order >= 3: cols += [u ** 3, v ** 3, u * u * v, u * v * v]
    return np.stack(cols, 1)

def fit_shading(I, A, cy, cx, order=2, iters=6):
    """iteratively reweighted fit on 'matrix' grains: drop grains far below the fit (phase/pores)."""
    D = design(cy, cx, order); w = A.copy(); keep = np.ones(len(I), bool)
    for _ in range(iters):
        W = np.sqrt(w[keep]); c, *_ = np.linalg.lstsq(D[keep] * W[:, None], I[keep] * W, rcond=None)
        r = I - D @ c; s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = r > -2.0 * max(s, 1.0)
    return c, D @ c, r, s, keep
