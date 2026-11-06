"""Segmentation-free (pixel-level) second-phase fraction estimators. Label-free; constants fixed a priori."""
import numpy as np
from scipy import ndimage as ndi

def _otsu(v, bins=128):
    h, e = np.histogram(v, bins=bins); c = (e[1:] + e[:-1]) / 2; w = h / h.sum(); mu = (w * c).sum(); var = (w * (c - mu) ** 2).sum()
    best, bk = -1, None
    cw = np.cumsum(w); cm = np.cumsum(w * c)
    for k in range(1, bins):
        w0 = cw[k - 1]; w1 = 1 - w0
        if w0 < 0.02 or w1 < 0.02: continue
        m0 = cm[k - 1] / w0; m1 = (cm[-1] - cm[k - 1]) / w1; b = w0 * w1 * (m0 - m1) ** 2
        if b > best: best, bk = b, k
    if bk is None: return np.nan, np.nan, np.nan, np.nan
    w0 = cw[bk - 1]; m0 = cm[bk - 1] / w0; m1 = (cm[-1] - cm[bk - 1]) / (1 - w0)
    return float(w0), float(best / var), float(m1 - m0), float(e[bk])

def _gmm2(v, iters=60):
    v = v.astype(np.float64); q = np.quantile(v, [0.15, 0.7]); mu = np.array(q); s = np.array([v.std() / 2] * 2); pi = np.array([0.3, 0.7])
    for _ in range(iters):
        ll = -0.5 * ((v[:, None] - mu) / s) ** 2 - np.log(s) + np.log(pi)
        ll -= ll.max(1, keepdims=True); r = np.exp(ll); r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9; pi = nk / nk.sum(); mu = (r * v[:, None]).sum(0) / nk; s = np.sqrt((r * (v[:, None] - mu) ** 2).sum(0) / nk) + 1e-3
    o = np.argsort(mu)
    return float(pi[o[0]]), float((mu[o[1]] - mu[o[0]]) / np.sqrt((s ** 2).mean()))

KEYS = []
for sg in (1.5, 2.5, 4.0):
    KEYS += [f'pp_otsu_frac_s{sg}', f'pp_otsu_eta_s{sg}', f'pp_otsu_gap_s{sg}', f'pp_gmm_frac_s{sg}', f'pp_gmm_sep_s{sg}', f'pp_mean_shift_s{sg}']

def features(im_u8):
    a = im_u8.astype(np.float64); out = {}
    for sg in (1.5, 2.5, 4.0):
        s = ndi.gaussian_filter(a, sg); v = s.ravel()
        # drop pores: darkest 2% and anything far below the low tail
        v = v[v > np.percentile(v, 2.0)]
        sub = v[::7]
        w0, eta, gap, thr = _otsu(v)
        out[f'pp_otsu_frac_s{sg}'] = w0; out[f'pp_otsu_eta_s{sg}'] = eta; out[f'pp_otsu_gap_s{sg}'] = gap
        gf, gs = _gmm2(sub); out[f'pp_gmm_frac_s{sg}'] = gf; out[f'pp_gmm_sep_s{sg}'] = gs
        hi = np.percentile(v, 75); lo = np.percentile(v, 5)
        out[f'pp_mean_shift_s{sg}'] = float((hi - v.mean()) / max(hi - lo, 1e-3))
    return out

def matrix(images):
    rows = [features(i) for i in images]
    return np.array([[r[k] for k in KEYS] for r in rows], float)
