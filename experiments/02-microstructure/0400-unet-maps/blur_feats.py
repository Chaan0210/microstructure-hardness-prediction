"""Per-image blur estimates (label-free).
psd: radial PSD; white-noise floor from the corner band; fit log(P-floor)+3log f = a - 4 pi^2 s^2 f^2 over the band where signal > 3x floor.
edge: pore edge 10-90 widths (median), NaN when no usable pore.
grad: boundary sharpness = (p99 of |grad| of sigma1 image) / (p95-p5 of sigma2 image)."""
import numpy as np
from scipy import ndimage as ndi
KEYS = ['bl_psd_sigma', 'bl_psd_r2', 'bl_hf_ratio', 'bl_edge_w', 'bl_edge_n', 'bl_grad_sharp']

def _psd(a):
    f = a - a.mean(); win = np.outer(np.hanning(256), np.hanning(256)); F = np.fft.fftshift(np.abs(np.fft.fft2(f * win)) ** 2)
    yy, xx = np.mgrid[-128:128, -128:128]; r = np.sqrt(yy ** 2 + xx ** 2).astype(int)
    P = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return P[:128], np.arange(128) / 256.0

def features(im_u8):
    a = im_u8.astype(np.float64); out = {}
    P, fr = _psd(a); floor = np.median(P[100:128])
    k = np.arange(4, 100); S = P[k] - floor; ok = S > 3 * floor
    if ok.sum() >= 6:
        x = fr[k][ok] ** 2; yv = np.log(S[ok]) + 3 * np.log(fr[k][ok])
        c = np.polyfit(x, yv, 1); pred = np.polyval(c, x)
        out['bl_psd_sigma'] = float(np.sqrt(max(-c[0], 0) / (4 * np.pi ** 2))); out['bl_psd_r2'] = float(1 - np.var(yv - pred) / max(np.var(yv), 1e-9))
    else:
        out['bl_psd_sigma'] = np.nan; out['bl_psd_r2'] = np.nan
    out['bl_hf_ratio'] = float((P[40:64].mean() - floor) / max(P[10:20].mean() - floor, 1e-9))
    # pore edges
    from skimage import measure
    s = ndi.gaussian_filter(a, 2.0); thr = np.percentile(s, 50) - 0.6 * (np.percentile(s, 50) - np.percentile(s, 0.2)); W = []
    yy, xx = np.mgrid[0:256, 0:256]
    for p in measure.regionprops(measure.label(s < thr)):
        if p.area < 20 or p.area > 400 or p.eccentricity > 0.8: continue
        cy, cx = p.centroid; R = np.sqrt(p.area / np.pi)
        if cy < R + 8 or cx < R + 8 or cy > 255 - R - 8 or cx > 255 - R - 8: continue
        y0, y1, x0, x1 = int(cy - R - 8), int(cy + R + 9), int(cx - R - 8), int(cx + R + 9)
        d = np.sqrt((yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2); sub = a[y0:y1, x0:x1]
        prof = np.array([sub[(d >= q - 0.5) & (d < q + 0.5)].mean() for q in np.arange(0, int(R) + 7)])
        inside = prof[:max(1, int(R * 0.5))].mean(); outside = prof[-3:].mean()
        if outside - inside < 20: continue
        f = (prof - inside) / (outside - inside); q = np.arange(len(f)); s0 = int(R * 0.3)
        try: W.append(np.interp(0.9, f[s0:], q[s0:]) - np.interp(0.1, f[s0:], q[s0:]))
        except Exception: pass
    W = np.array([w for w in W if 0 < w < 15])
    out['bl_edge_w'] = float(np.median(W)) if len(W) else np.nan; out['bl_edge_n'] = float(len(W))
    g1 = ndi.gaussian_filter(a, 1.0); gm = np.hypot(*np.gradient(g1)); s2 = ndi.gaussian_filter(a, 2.0)
    out['bl_grad_sharp'] = float(np.percentile(gm, 99) / max(np.percentile(s2, 95) - np.percentile(s2, 5), 1e-3))
    return out

def matrix(images):
    return np.array([[features(i)[k] for k in KEYS] for i in images], float)
