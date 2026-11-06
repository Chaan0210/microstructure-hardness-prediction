"""Candidate image-level statistics (label-free) using existing masks-v3: boundary contrast/width, grain-shade spread, illumination, sharpness."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

def extract(im_u8, mask):
    f = im_u8.astype(np.float64); out = {}
    sm = ndi.gaussian_filter(f, 0.7)
    q = np.quantile(ndi.gaussian_filter(f, 1.5), [0.005, 0.5, 0.95]); span = max(q[2] - q[0], 1e-3)
    bnd = np.zeros_like(mask, bool)
    bnd[:, :-1] |= mask[:, :-1] != mask[:, 1:]; bnd[:-1, :] |= mask[:-1, :] != mask[1:, :]
    near = ndi.binary_dilation(bnd, iterations=1); far = ~ndi.binary_dilation(bnd, iterations=3) & (mask > 0)
    if near.sum() > 50 and far.sum() > 50:
        out['bnd_contrast'] = float((np.median(sm[far]) - np.median(sm[near])) / span)
        d = ndi.distance_transform_edt(~bnd)
        prof = [np.median(sm[(d >= k - 0.5) & (d < k + 0.5) & (mask > 0)]) for k in range(0, 6)]
        out['bnd_prof1'] = float((prof[3] - prof[1]) / span); out['bnd_prof2'] = float((prof[4] - prof[2]) / span)
    else:
        out['bnd_contrast'] = np.nan; out['bnd_prof1'] = np.nan; out['bnd_prof2'] = np.nan
    props = measure.regionprops(mask, intensity_image=ndi.gaussian_filter(f, 1.0))
    I = np.array([p.intensity_mean for p in props if p.area > 20]); A = np.array([p.area for p in props if p.area > 20])
    if len(I) > 10:
        hi = np.quantile(I, 0.9); z = (hi - I) / span
        mt = z < np.quantile(z, 0.6)
        out['shade_std_matrix'] = float(np.std(I[mt]) / span); out['shade_iqr_all'] = float((np.quantile(I, 0.75) - np.quantile(I, 0.25)) / span)
    else:
        out['shade_std_matrix'] = np.nan; out['shade_iqr_all'] = np.nan
    il = ndi.gaussian_filter(f, 40, mode='reflect'); out['illum_std'] = float(il.std() / span)
    gy, gx = np.gradient(ndi.gaussian_filter(f, 1.0)); g = np.hypot(gx, gy); out['grad_q99'] = float(np.quantile(g, 0.99) / span)
    out['mean_int_rel'] = float((f.mean() - q[0]) / span)
    return out

def matrix(images, masks):
    rows = [extract(i, m) for i, m in zip(images, masks)]; keys = sorted(rows[0]); return np.array([[r[k] for k in keys] for r in rows]), keys
