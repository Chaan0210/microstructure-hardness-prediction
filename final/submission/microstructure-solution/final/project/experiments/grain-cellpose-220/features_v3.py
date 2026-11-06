import numpy as np
from scipy import ndimage as ndi
from skimage import measure, filters, morphology
from features import extract as extract_v2, _noise, _wq


def _smooth_field(val, w, sigma):
    num = ndi.gaussian_filter(val * w, sigma, mode='reflect'); den = ndi.gaussian_filter(w, sigma, mode='reflect')
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan)


def _pairs(mask):
    a = np.concatenate([mask[:, :-1].ravel(), mask[:-1, :].ravel()]); b = np.concatenate([mask[:, 1:].ravel(), mask[1:, :].ravel()])
    k = (a != b) & (a > 0) & (b > 0)
    p = np.unique(np.sort(np.stack([a[k], b[k]], 1), 1), axis=0)
    return p, a[k], b[k]


def extract_v3(im_u8, mask):
    base = extract_v2(im_u8, mask)
    if base is None:
        return None
    out = dict(base)
    f = im_u8.astype(np.float32) / 255; H, W = f.shape
    noise = _noise(f); sm = filters.gaussian(f, 1.0)
    inner = mask * (ndi.minimum_filter(mask, 3) == ndi.maximum_filter(mask, 3))
    props = {p.label: p for p in measure.regionprops(mask)}
    ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = np.array([l for l in props if l in ipr and ipr[l].area >= 3])
    A = np.array([props[l].area for l in labs], float)
    I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1.0) for l in labs])
    SOL = np.array([props[l].solidity for l in labs])
    hi = _wq(I, A, 0.75); lo = float(np.percentile(sm, 0.1)); span = max(hi - lo, 1e-3)
    z = (hi - I) / span; medA = np.median(A)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3.0 * medA)
    g = ~pore; sp = g & (z > 0.3)
    eqd = np.sqrt(4 * A / np.pi)
    Ag, dg = A[g], eqd[g]
    # --- size distribution (area-weighted, tails, shape) ---
    out['aw_eqd_mean'] = float(np.sum(Ag * dg) / Ag.sum())
    out['aw_inv_sqrt_d'] = float(np.sum(Ag * dg ** -0.5) / Ag.sum())
    out['eqd_q10'] = float(np.quantile(dg, 0.1)); out['eqd_q90'] = float(np.quantile(dg, 0.9))
    out['aw_eqd_q90'] = _wq(dg, Ag, 0.9); out['aw_eqd_q50'] = _wq(dg, Ag, 0.5)
    md = np.median(dg)
    out['coarse_area_frac'] = float(Ag[dg > 2 * md].sum() / Ag.sum())
    out['fine_area_frac'] = float(Ag[dg < 0.5 * md].sum() / Ag.sum())
    la = np.log(Ag); n = len(la)
    if n > 3 and la.std() > 0:
        s = float(((la - la.mean()) ** 3).mean() / la.std() ** 3); k = float(((la - la.mean()) ** 4).mean() / la.std() ** 4 - 3)
        out['logA_skew'] = s; out['logA_bimodality'] = float((s ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))
    else:
        out['logA_skew'] = 0.0; out['logA_bimodality'] = 0.0
    # --- spatial grain-size field ---
    la_map = np.zeros((H, W)); wmap = np.zeros((H, W))
    lut_la = np.zeros(mask.max() + 1); lut_w = np.zeros(mask.max() + 1)
    lut_la[labs[g]] = np.log(A[g]); lut_w[labs[g]] = 1.0
    la_map = lut_la[mask]; wmap = lut_w[mask]
    for s in (16, 32):
        S = _smooth_field(la_map, wmap, s); v = S[np.isfinite(S)]
        out[f'sizefield_std_s{s}'] = float(v.std()) if v.size else 0.0
        if s == 32 and v.size:
            out['sizefield_range_s32'] = float(np.quantile(v, 0.95) - np.quantile(v, 0.05))
            yy, xx = np.mgrid[0:H, 0:W]; ok = np.isfinite(S)
            Xd = np.c_[np.ones(ok.sum()), yy[ok], xx[ok]]; coef, *_ = np.linalg.lstsq(Xd, S[ok], rcond=None)
            res = S[ok] - Xd @ coef; out['sizefield_plane_r2'] = float(1 - res.var() / max(S[ok].var(), 1e-9))
    if 'sizefield_range_s32' not in out:
        out['sizefield_range_s32'] = 0.0; out['sizefield_plane_r2'] = 0.0
    out['sizefield_ratio'] = out['sizefield_std_s16'] / max(out['sizefield_std_s32'], 1e-6)
    # 8x8 tile std of area-weighted log size
    vals = []
    for a in range(8):
        for b in range(8):
            t = mask[a * 32:(a + 1) * 32, b * 32:(b + 1) * 32]; tw = wmap[a * 32:(a + 1) * 32, b * 32:(b + 1) * 32]
            if tw.sum() > 200: vals.append((la_map[a * 32:(a + 1) * 32, b * 32:(b + 1) * 32] * tw).sum() / tw.sum())
    out['tile8_logA_std'] = float(np.std(vals)) if len(vals) > 8 else 0.0
    # --- second phase: number fraction, darkness, contiguity, clustering ---
    out['sp_count_frac'] = float(sp.sum() / max(g.sum(), 1))
    out['sp_z_mean'] = float(z[sp].mean()) if sp.any() else 0.0
    pairs, _, _ = _pairs(mask)
    is_sp = np.zeros(mask.max() + 1, bool); is_sp[labs[sp]] = True
    is_g = np.zeros(mask.max() + 1, bool); is_g[labs[g]] = True
    if len(pairs):
        gp = pairs[is_g[pairs[:, 0]] & is_g[pairs[:, 1]]]
        ss = (is_sp[gp[:, 0]] & is_sp[gp[:, 1]]).sum(); sm_ = (is_sp[gp[:, 0]] ^ is_sp[gp[:, 1]]).sum()
        out['sp_contiguity'] = float(2 * ss / max(2 * ss + sm_, 1))
        deg = np.bincount(gp.ravel(), minlength=mask.max() + 1)[labs[g]]
        out['neighbors_mean'] = float(deg.mean()); out['neighbors_std'] = float(deg.std())
    else:
        out['sp_contiguity'] = 0.0; out['neighbors_mean'] = 0.0; out['neighbors_std'] = 0.0
    spmap = is_sp[mask].astype(float); S = _smooth_field(spmap, (mask > 0).astype(float), 32); v = S[np.isfinite(S)]
    out['sp_field_std_s32'] = float(v.std()) if v.size else 0.0
    # --- segmentation-independent pores (pixel level, affine-invariant scale) ---
    sig = max(1.5, 40 * noise)
    zp = (hi - filters.gaussian(f, sig)) / span
    pm = morphology.opening(zp > 0.75, morphology.disk(1))
    pr = [r for r in measure.regionprops(measure.label(pm)) if r.area >= 6]
    out['px_pore_n'] = float(len(pr)); out['px_pore_frac'] = float(sum(r.area for r in pr) / (H * W))
    pdm = float(np.mean([r.equivalent_diameter_area for r in pr])) if pr else 0.0
    out['px_pore_d'] = pdm; out['px_pore_ecc'] = float(np.mean([r.eccentricity for r in pr])) if pr else 0.0
    out['pore_to_grain_size'] = pdm / max(out['aw_eqd_mean'], 1e-3)
    # --- boundary-map alignment (structure tensor on mask boundaries) ---
    bnd = (mask[:, :-1] != mask[:, 1:])[:-1, :] | (mask[:-1, :] != mask[1:, :])[:, :-1]
    bs = filters.gaussian(bnd.astype(float), 1.0); gy, gx = np.gradient(bs)
    jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    out['boundary_coherence'] = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / max(jxx + jyy, 1e-9))
    return out


def feature_matrix_v3(images, masks):
    rows = [extract_v3(i, m) for i, m in zip(images, masks)]
    keys = sorted({k for r in rows if r for k in r})
    X = np.array([[r.get(k, np.nan) if r else np.nan for k in keys] for r in rows], float)
    return X, keys
