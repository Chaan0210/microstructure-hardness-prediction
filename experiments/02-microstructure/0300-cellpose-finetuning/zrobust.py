"""Noise-robust intensity-normalised (z) features. features.py sets the dark reference lo = 0.1th percentile of gaussian(f, 1) pixels,
which drifts darker with image noise (residual-noise tail), inflating the span and shrinking every z -> fewer second-phase grains.
Here lo comes from object interior MEANS (area-weighted 1st percentile of per-object mean intensity), which averages the noise away.
Same object table / pore & second-phase rules as features.py; returns extra columns with suffix _r."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, filters

def _wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum(); return float(x[o][min(np.searchsorted(c, q), len(x) - 1)])

KEYS = ['lo_r', 'span_ratio_r', 'f_dark_20_r', 'f_dark_35_r', 'f_dark_50_r', 'z_q50_r', 'z_q75_r', 'z_q90_r', 'pore_n_r', 'pore_frac_r',
        'sp_frac_grains_r', 'sp_count_frac_r', 'sp_z_mean_r', 'tile_spfrac_std_r']

def extract_r(im_u8, mask):
    f = im_u8.astype(np.float32) / 255; H, W = f.shape; sm = filters.gaussian(f, 1.0)
    inner = mask * (ndi.minimum_filter(mask, 3) == ndi.maximum_filter(mask, 3))
    props = {p.label: p for p in measure.regionprops(mask)}
    rows = []
    for q in measure.regionprops(inner, intensity_image=sm):
        p = props.get(q.label)
        if p is None or q.area < 3: continue
        rows.append((p.area, p.axis_major_length / max(p.axis_minor_length, 1.0), p.solidity, q.intensity_mean, *p.centroid))
    if len(rows) < 5: return None
    A, AR, SOL, I, CY, CX = [np.array(c, float) for c in zip(*rows)]
    hi = _wq(I, A, 0.75); lo = _wq(I, A, 0.01); lo_px = float(np.percentile(sm, 0.1))
    span = max(hi - lo, 1e-3); z = (hi - I) / span
    out = {'lo_r': lo, 'span_ratio_r': span / max(hi - lo_px, 1e-3)}
    for t in (0.2, 0.35, 0.5): out[f'f_dark_{int(t * 100)}_r'] = float(A[z > t].sum() / A.sum())
    for qq in (0.5, 0.75, 0.9): out[f'z_q{int(qq * 100)}_r'] = _wq(z, A, qq)
    medA = np.median(A); pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3.0 * medA); g = ~pore; sp = g & (z > 0.3)
    out['pore_n_r'] = float(pore.sum()); out['pore_frac_r'] = float(A[pore].sum() / (H * W))
    out['sp_frac_grains_r'] = float(A[sp].sum() / A[g].sum()); out['sp_count_frac_r'] = float(sp.sum() / max(g.sum(), 1))
    out['sp_z_mean_r'] = float(z[sp].mean()) if sp.any() else 0.0
    ty = np.minimum((CY // 64).astype(int), 3); tx = np.minimum((CX // 64).astype(int), 3); fd = []
    for a in range(4):
        for b in range(4):
            s = g & (ty == a) & (tx == b)
            if s.sum() >= 2: fd.append(A[s & sp].sum() / A[s].sum())
    out['tile_spfrac_std_r'] = float(np.std(fd)) if len(fd) > 3 else 0.0
    return out
