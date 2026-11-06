"""Segmentation-free, noise-robust texture-scale features (label-free, fixed constants).
White noise only affects autocorrelation at lag 0; blur mostly lags <= 2. Scale = lag where AC(r)/AC(2) falls to 0.5 / 0.25."""
import numpy as np
from scipy import ndimage as ndi

def _ac2d(f):
    f = f - f.mean(); F = np.fft.fft2(f, s=(2 * f.shape[0], 2 * f.shape[1]))
    ac = np.real(np.fft.ifft2(np.abs(F) ** 2)); ac = np.fft.fftshift(ac)
    n = np.real(np.fft.ifft2(np.abs(np.fft.fft2(np.ones_like(f), s=ac.shape)) ** 2)); n = np.fft.fftshift(n)
    return ac / np.maximum(n, 1)

def _radial(ac, rmax):
    c = np.array(ac.shape) // 2; yy, xx = np.indices(ac.shape); r = np.hypot(yy - c[0], xx - c[1])
    th = np.arctan2(yy - c[0], xx - c[1])
    prof = np.array([ac[(r >= k - 0.5) & (r < k + 0.5)].mean() for k in range(rmax + 1)])
    return prof, r, th

def _cross(prof, base, lev):
    p = prof / base
    for k in range(3, len(p)):
        if p[k] < lev:
            return (k - 1) + (p[k - 1] - lev) / max(p[k - 1] - p[k], 1e-9)
    return float(len(p))

def scale_of(f, rmax=48):
    ac = _ac2d(f); prof, r, th = _radial(ac, rmax); b = prof[2]
    if b <= 0: return np.nan, np.nan, prof
    return _cross(prof, b, 0.5), _cross(prof, b, 0.25), prof

def extract(im_u8):
    f = im_u8.astype(np.float64) / 255
    f = f - ndi.gaussian_filter(f, 32, mode='reflect')       # remove illumination drift
    out = {}
    L50, L25, prof = scale_of(f)
    out['ac_L50'] = L50; out['ac_L25'] = L25
    for k in (4, 8, 16, 24, 32):
        out[f'ac_r{k}'] = prof[k] / prof[2] if prof[2] > 0 else np.nan
    # anisotropy at lag 8 and 16 from the 2D AC
    ac = _ac2d(f); c = np.array(ac.shape) // 2; yy, xx = np.indices(ac.shape); r = np.hypot(yy - c[0], xx - c[1]); th = np.arctan2(yy - c[0], xx - c[1])
    for k in (6, 12):
        ring = (r >= k - 0.5) & (r < k + 0.5); v = ac[ring] / max(ac[c[0], c[1] + 2], 1e-12); t = th[ring]
        z = np.sum(v * np.exp(2j * t)) / max(np.sum(np.abs(v)), 1e-12)
        out[f'ac_aniso{k}'] = float(np.abs(z))
        bins = np.floor((t % np.pi) / (np.pi / 8)).astype(int); m = np.array([v[bins == b].mean() for b in range(8)])
        out[f'ac_maxmin{k}'] = float(m.max() - m.min())
    # local scale field on 4x4 and 8x8 tiles
    for nt in (4, 8):
        s = 256 // nt; Ls = []
        for a in range(nt):
            for b in range(nt):
                t = f[a * s:(a + 1) * s, b * s:(b + 1) * s]
                l50, _, _ = scale_of(t, rmax=min(s // 2, 32)); Ls.append(l50)
        Ls = np.log(np.clip(np.array(Ls, float), 1, None)).reshape(nt, nt)
        out[f'tile{nt}_logL_std'] = float(np.nanstd(Ls))
        out[f'tile{nt}_logL_range'] = float(np.nanquantile(Ls, 0.9) - np.nanquantile(Ls, 0.1))
        if nt == 8:
            yy8, xx8 = np.mgrid[0:nt, 0:nt]; ok = np.isfinite(Ls)
            A = np.c_[np.ones(ok.sum()), yy8[ok], xx8[ok]]; coef, *_ = np.linalg.lstsq(A, Ls[ok], rcond=None)
            res = Ls[ok] - A @ coef; out['tile8_logL_plane_r2'] = float(1 - res.var() / max(Ls[ok].var(), 1e-12)); out['tile8_logL_grad'] = float(np.hypot(coef[1], coef[2]))
    # smoothed intensity distribution (phase contrast), affine-invariant
    sm = ndi.gaussian_filter(im_u8.astype(np.float64), 2.0)
    q = np.quantile(sm, [0.02, 0.1, 0.25, 0.5, 0.75, 0.9]); span = max(q[5] - q[0], 1e-6)
    out['sm_q_skew'] = float((q[5] + q[0] - 2 * q[3]) / span); out['sm_q_lowtail'] = float((q[3] - q[1]) / span); out['sm_q_iqr'] = float((q[4] - q[2]) / span)
    return out

def matrix(images):
    rows = [extract(i) for i in images]; keys = sorted(rows[0]); return np.array([[r[k] for k in keys] for r in rows]), keys
