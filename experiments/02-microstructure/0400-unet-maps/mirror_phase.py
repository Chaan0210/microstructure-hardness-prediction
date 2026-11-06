"""Mirror-subtraction second-phase estimator on smoothed pixels: matrix mode assumed symmetric; excess dark mass = phase (+pores, boundaries)."""
import numpy as np
from scipy import ndimage as ndi
SIG = (1.5, 2.5, 3.5)
KS = (0.0, 1.0, 2.0, 3.0)
KEYS = [f'mp_ex_k{k:g}_s{s:g}' for s in SIG for k in KS] + [f'mp_sr_s{s:g}' for s in SIG] + [f'mp_lmean_s{s:g}' for s in SIG]

def _mode(v):
    h, e = np.histogram(v, bins=np.arange(np.floor(v.min()), np.ceil(v.max()) + 1, 0.5))
    hs = ndi.gaussian_filter1d(h.astype(float), 4); k = np.argmax(hs); return (e[k] + e[k + 1]) / 2

def features(im_u8):
    a = im_u8.astype(np.float64); out = {}
    for s in SIG:
        v = ndi.gaussian_filter(a, s).ravel(); m = _mode(v); r = v[v > m] - m; sr = np.sqrt(np.mean(r ** 2)) if len(r) else 1.0
        out[f'mp_sr_s{s:g}'] = float(sr)
        for k in KS:
            out[f'mp_ex_k{k:g}_s{s:g}'] = float(np.mean(v < m - k * sr) - np.mean(v > m + k * sr))
        # mean darkness of excess dark pixels relative to spread
        d = v[v < m - 2 * sr]; out[f'mp_lmean_s{s:g}'] = float((m - d.mean()) / sr) if len(d) > 50 else 0.0
    return out

def matrix(images):
    rows = [features(i) for i in images]
    return np.array([[r[k] for k in KEYS] for r in rows], float)
