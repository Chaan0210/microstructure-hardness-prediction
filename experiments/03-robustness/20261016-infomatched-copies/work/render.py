"""Information-matched rendering of a CLEAN train image toward the real noisy (noise >= 12) preset. Only permitted operations (user's DACON Q&A,
memory microstructure-compliance-1400): brightness/contrast changes of analysis-selected regions (grains from the image's own Cellpose mask) of the
same image; replacing boundary pixels with surrounding pixels of the same image; global blur, noise, contrast and brightness. No other image,
no test data, no labels.

Steps (all parameters per image drawn from distributions whose hyper-parameters are calibrated in calib.py on pixel statistics of REAL noisy train
images only):
 1. grain interior mean I over pixels >= 2 px from any label change; robust quadratic bright-grain shading S (as push-900/realdeg2); z = 1 - I/S.
 2. per-grain contrast: dark grains (z > 0.12, not pores) dev*c_d with c_d ~ clip(N(mu_img, CD_SD), 0, 1.3), mu_img ~ N(MU_D, SD_MUD) (phase-contrast
    compression toward the matrix); pores dev*k_p; matrix grains E_MAT*dev + N(0, J)*S with J ~ |N(J_MU, J_SD)| (grain-to-grain spread).
 3. boundary erasure: every pixel within BW px of a label change gets its own grain's new mean + white texture of the clean noise level, applied
    with probability P per grain (P_LO..P_HI per image; 1 = complete erasure).
 4. global brightness: the image mean is restored to the source mean (real noisy and clean bands share mean 145; erasure brightens by ~4 gray).
 5. global blur sigma ~ U(B_LO, B_HI); quadratic shading U(0, SHADE_MAX); optional global low-frequency noise field (MOT: sd as a fraction of S,
    Gaussian-correlated at MOT_S px); white noise to the target total noise (Haar-HH measured) drawn from the real noisy pool."""
import numpy as np
from scipy import ndimage as ndi
N = 256; YY, XX = np.mgrid[0:N, 0:N] / 255.0 - 0.5
BASE = dict(MU_D=0.675, SD_MUD=0.05, CD_SD=0.3, PORE_K=0.9, PORE_SD=0.05, E_MAT=1.0, J_MU=0.027, J_SD=0.008, BW=1, P_LO=0.8, P_HI=1.0,
            KEEP_MEAN=0.0, B_LO=0.75, B_HI=1.05, SHADE_MAX=0.05, MOT=0.0, MOT_S=3.0)

def hh(f):
    h = (f[0::2, 0::2] - f[0::2, 1::2] - f[1::2, 0::2] + f[1::2, 1::2]) / 2; return float(np.median(np.abs(h)) / 0.6745)

def render(a, lab, r, prm, noise_target):
    p = dict(BASE, **prm); a = a.astype(np.float64); m = lab.astype(np.int64); n = int(m.max()) + 1; f = a.copy(); sd0 = hh(a)
    edge = (ndi.minimum_filter(m, 3) != ndi.maximum_filter(m, 3)) | (m == 0)
    dist = ndi.distance_transform_edt(~edge)
    deep = np.where(dist >= 2, m, 0)
    cnt = np.bincount(deep.ravel(), minlength=n).astype(float); w = np.maximum(cnt, 1)
    I = np.bincount(deep.ravel(), weights=a.ravel(), minlength=n) / w
    cy = np.bincount(deep.ravel(), weights=YY.ravel(), minlength=n) / w; cx = np.bincount(deep.ravel(), weights=XX.ravel(), minlength=n) / w
    ok = cnt >= 6; ok[0] = False
    lab2 = np.where(m > 0, m, ndi.grey_dilation(m, size=3)) if (m == 0).any() else m
    if ok.sum() >= 10:
        D = np.c_[np.ones(n), cx, cy, cx * cx, cy * cy, cx * cy]; keep = ok & (I > np.quantile(I[ok], 0.4))
        for _ in range(4):
            sw = np.sqrt(cnt[keep]); b = np.linalg.lstsq(D[keep] * sw[:, None], I[keep] * sw, rcond=None)[0]; S = D @ b; keep = ok & (I / S > 0.95)
        z = 1 - I / S; dev = I - S
        pore = ok & (z > 0.42) & (cnt < 3 * np.median(cnt[ok])); dark = ok & (z > 0.12) & ~pore
        cd = np.clip(r.normal(r.normal(p['MU_D'], p['SD_MUD']), p['CD_SD'], n), 0, 1.3); J = abs(r.normal(p['J_MU'], p['J_SD']))
        newdev = np.where(dark, cd * dev, np.where(pore, r.normal(p['PORE_K'], p['PORE_SD']) * dev, p['E_MAT'] * dev + r.normal(0, J, n) * S))
        shift = np.where(ok, newdev - dev, 0.0); f = f + shift[lab2]
        if p['P_HI'] > 0:
            band = (dist <= p['BW']) & ok[lab2]
            drop = r.rand(n) < r.uniform(p['P_LO'], p['P_HI']); sel = band & drop[lab2]
            f[sel] = (I + shift)[lab2][sel] + r.normal(0, sd0, int(sel.sum()))
        Smed = float(np.median(S[ok]))
    else:
        Smed = float(np.median(a))
    if p['KEEP_MEAN'] > 0: f = f + p['KEEP_MEAN'] * (a.mean() - f.mean())
    bx = r.uniform(p['B_LO'], p['B_HI'])
    if bx > 0.05: f = ndi.gaussian_filter(f, bx)
    cq = r.normal(0, 1, 5); sh = cq[0] * YY + cq[1] * XX + cq[2] * YY * YY + cq[3] * XX * XX + cq[4] * XX * YY
    f = f + r.uniform(0, p['SHADE_MAX']) * Smed * sh / max(sh.std(), 1e-6)
    if p['MOT'] > 0:
        g = ndi.gaussian_filter(r.normal(0, 1, f.shape), p['MOT_S']); f = f + p['MOT'] * Smed * g / g.std()
    add = float(np.sqrt(max(noise_target ** 2 - hh(f) ** 2, 1.0)))
    return np.clip(np.round(f + r.normal(0, add, f.shape)), 0, 255)
