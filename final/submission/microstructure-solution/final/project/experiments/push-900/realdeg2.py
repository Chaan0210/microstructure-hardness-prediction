import numpy as np
from scipy import ndimage as ndi
YY, XX = np.mgrid[0:256, 0:256] / 255.0 - 0.5

DEFAULT = dict(MU_D=0.675, SD_MUD=0.05, CD_SD=0.3, PORE_K=0.9, PORE_SD=0.05, J_MU=0.027, J_SD=0.008, LINE_MODE='patchy', P_LO=0.28, P_HI=0.68,
               LINE_K=0.5, BLUR_MU=0.75, BLUR_SD=0.15, SHADE_MAX=0.05, NOISE_POOL=(12.0, 20.5), E_MAT=1.0)
LIGHT = dict(MU_D=1.0, SD_MUD=0.0, CD_SD=0.0, PORE_K=1.0, PORE_SD=0.0, J_MU=0.0, J_SD=0.0, P_LO=0.0, P_HI=0.0, BLUR_MU=0.0, BLUR_SD=0.0, SHADE_MAX=0.0)
MID_STRUCT_P = 0.62

def degrade_noisy_preset(a, lab, r, noise_pool):
    sdn = float(r.choice(noise_pool)); prm = None if (sdn >= 12 or r.rand() < MID_STRUCT_P) else LIGHT
    return degrade2(a, lab, r, prm, noise=sdn)

def degrade2(a, lab, r, prm=None, noise=None):
    p = dict(DEFAULT, **(prm or {})); m = lab.astype(np.int64); n = int(m.max()) + 1; f = a.astype(np.float64).copy()
    er = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    cnt = np.bincount(er.ravel(), minlength=n).astype(float); w = np.maximum(cnt, 1)
    I = np.bincount(er.ravel(), weights=f.ravel(), minlength=n) / w
    cy = np.bincount(er.ravel(), weights=YY.ravel(), minlength=n) / w; cx = np.bincount(er.ravel(), weights=XX.ravel(), minlength=n) / w
    ok = cnt >= 8; ok[0] = False
    sd_noise = noise if noise is not None else (r.uniform(*p['NOISE_POOL']) if isinstance(p['NOISE_POOL'], tuple) else float(r.choice(p['NOISE_POOL'])))
    if ok.sum() >= 10:
        D = np.c_[np.ones(n), cx, cy, cx * cx, cy * cy, cx * cy]; keep = ok & (I > np.quantile(I[ok], 0.4))
        for _ in range(4):
            sw = np.sqrt(cnt[keep]); b = np.linalg.lstsq(D[keep] * sw[:, None], I[keep] * sw, rcond=None)[0]; S = D @ b; keep = ok & (I / S > 0.95)
        z = 1 - I / S; dev = I - S
        pore = ok & (z > 0.42) & (cnt < 3 * np.median(cnt[ok])); dark = ok & (z > 0.12) & ~pore
        cd = np.clip(r.normal(r.normal(p['MU_D'], p['SD_MUD']), p['CD_SD'], n), 0, 1.3); J = abs(r.normal(p['J_MU'], p['J_SD']))
        newdev = np.where(dark, cd * dev, np.where(pore, r.normal(p['PORE_K'], p['PORE_SD']) * dev, p['E_MAT'] * dev + r.normal(0, J, n) * S))
        shift = np.where(ok, newdev - dev, 0.0); f = f + shift[m]
        lab2 = np.where(m > 0, m, ndi.grey_dilation(m, size=3)) if (m == 0).any() else m
        bnd = (m == 0) | (ndi.minimum_filter(m, 3) != ndi.maximum_filter(m, 3)); base = (I + shift)[lab2]
        if p['LINE_MODE'] == 'patchy':
            drop = r.rand(n) < r.uniform(p['P_LO'], p['P_HI']); sel = bnd & drop[lab2] & ok[lab2]; f[sel] = base[sel] + r.normal(0, 1.0, int(sel.sum()))
        elif p['LINE_MODE'] == 'pair':   # whole boundaries (both sides) vanish with p ~ U(P_LO, P_HI), keyed by the adjacent grain pair
            mx, mn = ndi.maximum_filter(lab2, 3), ndi.minimum_filter(lab2, 3); oth = np.where(mx != lab2, mx, mn)
            key = (np.minimum(lab2, oth) * n + np.maximum(lab2, oth)).astype(np.float64); seed = r.uniform(0, 1000)
            u = np.abs(np.sin(key * 12.9898 + seed) * 43758.5453) % 1.0; sel = bnd & (u < r.uniform(p['P_LO'], p['P_HI'])) & ok[lab2]
            f[sel] = base[sel] + r.normal(0, 1.0, int(sel.sum()))
        else:
            sel = bnd & ok[lab2]; f[sel] = base[sel] + p['LINE_K'] * (f[sel] - base[sel])
        Smed = float(np.median(S[ok]))
    else:
        Smed = float(np.median(f))
    bx = abs(r.normal(p['BLUR_MU'], p['BLUR_SD']))
    if bx > 0.05: f = ndi.gaussian_filter(f, bx)
    cq = r.normal(0, 1, 5); sh = cq[0] * YY + cq[1] * XX + cq[2] * YY * YY + cq[3] * XX * XX + cq[4] * XX * YY
    f = f + r.uniform(0, p['SHADE_MAX']) * Smed * sh / max(sh.std(), 1e-6)
    return np.clip(np.round(f + r.normal(0, sd_noise, f.shape)), 0, 255)
