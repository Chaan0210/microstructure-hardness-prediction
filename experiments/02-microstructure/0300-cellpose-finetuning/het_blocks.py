"""Block-wise grain-size heterogeneity from a label mask (idea credited to a public participant note; implementation written here).
Non-pore grains (same pore rule as features.py) are assigned to G x G blocks by centroid; per block b = -2*log(mean(a^-1/2)) over grain
areas a (px); het_G = std of b over blocks with >= 3 grains; N_eff = interior grains + 0.5 * border grains. Also the whole-image b."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, filters

def _wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum(); return float(x[o][min(np.searchsorted(c, q), len(x) - 1)])

KEYS = ['het4_b', 'het8_b', 'het4_range_b', 'neff', 'neff_q', 'b_all']

def extract_het(im_u8, mask):
    f = im_u8.astype(np.float32) / 255; H, W = f.shape; sm = filters.gaussian(f, 1.0)
    inner = mask * (ndi.minimum_filter(mask, 3) == ndi.maximum_filter(mask, 3))
    props = {p.label: p for p in measure.regionprops(mask)}; rows = []
    for q in measure.regionprops(inner, intensity_image=sm):
        p = props.get(q.label)
        if p is None or q.area < 3: continue
        r0, c0, r1, c1 = p.bbox
        rows.append((p.area, p.axis_major_length / max(p.axis_minor_length, 1.0), p.solidity, q.intensity_mean, *p.centroid, r0 == 0 or c0 == 0 or r1 == H or c1 == W))
    if len(rows) < 5: return None
    A, AR, SOL, I, CY, CX, B = [np.array(c, float) for c in zip(*rows)]; B = B.astype(bool)
    hi = _wq(I, A, 0.75); lo = float(np.percentile(sm, 0.1)); z = (hi - I) / max(hi - lo, 1e-3)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3.0 * np.median(A)); g = ~pore
    out = {'neff': float((g & ~B).sum() + 0.5 * (g & B).sum())}; out['neff_q'] = out['neff'] ** 0.25
    out['b_all'] = float(-2 * np.log(np.mean(A[g] ** -0.5)))
    for G in (4, 8):
        ty = np.minimum((CY // (H / G)).astype(int), G - 1); tx = np.minimum((CX // (W / G)).astype(int), G - 1); bs = []
        for a in range(G):
            for b in range(G):
                s = g & (ty == a) & (tx == b)
                if s.sum() >= 3: bs.append(-2 * np.log(np.mean(A[s] ** -0.5)))
        out[f'het{G}_b'] = float(np.std(bs)) if len(bs) > 3 else np.nan
        if G == 4: out['het4_range_b'] = float(np.max(bs) - np.min(bs)) if len(bs) > 3 else np.nan
    return out
