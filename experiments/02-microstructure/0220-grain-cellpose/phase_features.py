"""Second-phase estimators (label-free). Chosen on synthetic images with known truth (phase_synth_test.py), not on hardness labels."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, filters
from sklearn.mixture import GaussianMixture


def grain_table(im_u8, mask):
    f = im_u8.astype(np.float32) / 255; sm = filters.gaussian(f, 1.0)
    inner = mask * (ndi.minimum_filter(mask, 3) == ndi.maximum_filter(mask, 3))
    props = {p.label: p for p in measure.regionprops(mask)}
    rows = []
    for q in measure.regionprops(inner, intensity_image=sm):
        p = props[q.label]
        if q.area < 3: continue
        rows.append((q.label, p.area, q.intensity_mean, q.area, p.solidity, p.axis_major_length / max(p.axis_minor_length, 1.0)))
    if len(rows) < 5: return None
    L, A, I, Ai, SOL, AR = [np.array(c, float) for c in zip(*rows)]
    return dict(label=L.astype(int), A=A, I=I, Ainner=Ai, SOL=SOL, AR=AR, sm=sm)


def _mode(x, w=None):
    lo, hi = np.quantile(x, [0.01, 0.99]); grid = np.linspace(lo, hi, 256)
    bw = 1.06 * np.std(x) * len(x) ** -0.2 + 1e-6
    dens = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2) @ (np.ones(len(x)) if w is None else w)
    return grid[np.argmax(dens)]


def phase_estimates(t):
    """t: grain_table output. Returns dict of second-phase estimators (dark = second phase)."""
    A, I, SOL, AR = t['A'], t['I'], t['SOL'], t['AR']
    out = {}
    # robust dark reference: interior means of compact objects in the darkest 3% (pores), fallback to darkest object
    order = np.argsort(I); k = max(1, int(0.03 * len(I)))
    cand = order[:k]; cand = cand[(SOL[cand] > 0.85) & (AR[cand] < 2.5)] if np.any((SOL[cand] > 0.85) & (AR[cand] < 2.5)) else order[:1]
    lo = float(np.median(I[cand]))
    m0 = float(_mode(I))                                   # matrix level = dominant mode of grain means
    span = max(m0 - lo, 1e-3)
    z = (m0 - I) / span                                    # 0 = matrix mode, 1 = pore level
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3.0 * np.median(A))
    g = ~pore; zg, Ag = z[g], A[g]
    # matrix spread from the bright (left) half around the mode -> robust sigma
    left = zg[zg <= 0]; s = float(np.sqrt(np.mean(left ** 2))) if left.size >= 5 else float(np.std(zg))
    out['ph_matrix_sigma'] = s
    # fixed thresholds in the new scale (count and area)
    for th in (0.12, 0.18, 0.25):
        out[f'ph_count_z{int(th*100)}'] = float((zg > th).mean()); out[f'ph_area_z{int(th*100)}'] = float(Ag[zg > th].sum() / Ag.sum())
    # mirror-excess: dark tail minus mirrored bright tail (assumes symmetric matrix spread)
    for c in (1.5, 2.5):
        d = c * s
        out[f'ph_mirror_count_{int(c*10)}'] = float(max(((zg > d).sum() - (zg < -d).sum()) / len(zg), 0.0))
        out[f'ph_mirror_area_{int(c*10)}'] = float(max((Ag[zg > d].sum() - Ag[zg < -d].sum()) / Ag.sum(), 0.0))
    # 2-component GMM on grain means (affine-equivariant); soft count/area fractions of the darker component
    x = zg.reshape(-1, 1)
    try:
        g1 = GaussianMixture(1, random_state=0).fit(x); g2 = GaussianMixture(2, random_state=0, n_init=3).fit(x)
        dark = int(np.argmax(g2.means_.ravel())); post = g2.predict_proba(x)[:, dark]
        out['ph_gmm_count'] = float(post.mean()); out['ph_gmm_area'] = float(np.sum(post * Ag) / Ag.sum())
        mu = g2.means_.ravel(); sd = np.sqrt(g2.covariances_.ravel())
        out['ph_gmm_sep'] = float(abs(mu[1] - mu[0]) / np.sqrt(sd[0] * sd[1] + 1e-9))
        out['ph_gmm_dbic'] = float(g1.bic(x) - g2.bic(x)) / len(x)
        out['ph_gmm_dark_mu'] = float(mu[dark])
    except Exception:
        out.update(ph_gmm_count=0.0, ph_gmm_area=0.0, ph_gmm_sep=0.0, ph_gmm_dbic=0.0, ph_gmm_dark_mu=0.0)
    return out
