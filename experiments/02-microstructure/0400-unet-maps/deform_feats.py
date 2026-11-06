"""Label-free deformation / size-gradient features from instance masks.
Shape tensor per grain: second central moments / sqrt(det) (unit determinant). Averaging tensors over many grains cancels the random
anisotropy of individual Voronoi cells and leaves the imposed stretch. Size gradient: grain-level OLS of log area on position."""
import numpy as np
from scipy import ndimage as ndi

KEYS = ['df_glob_lar', 'df_loc64_lar_mean', 'df_loc64_lar_std', 'df_loc32_lar_mean', 'df_loc32_lar_std', 'df_dir_coh64', 'df_grain_lar_wmean',
        'df_grain_lar_q75', 'sg_slope', 'sg_r2', 'sg_quad_r2', 'sg_loc64_std', 'df_lar_vs_size_corr', 'df_stretch_x_size']

def _tensor_lar(Sxx, Syy, Sxy):
    tr = Sxx + Syy; d = np.sqrt(np.maximum((Sxx - Syy) ** 2 + 4 * Sxy ** 2, 0))
    l1, l2 = (tr + d) / 2, np.maximum((tr - d) / 2, 1e-9)
    return 0.5 * np.log(l1 / l2)

def features(mask, intensity=None, pore_like=None):
    m = mask.astype(np.int64); H, W = m.shape
    lab = np.unique(m); lab = lab[lab > 0]
    if len(lab) < 20: return {k: np.nan for k in KEYS}
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    A = ndi.sum(np.ones_like(yy), m, lab); cy = ndi.sum(yy, m, lab) / A; cx = ndi.sum(xx, m, lab) / A
    myy = ndi.sum(yy * yy, m, lab) / A - cy ** 2; mxx = ndi.sum(xx * xx, m, lab) / A - cx ** 2; mxy = ndi.sum(xx * yy, m, lab) / A - cx * cy
    # border grains: drop (truncated shapes)
    bmask = np.zeros(m.max() + 1, bool); bmask[np.unique(np.r_[m[0], m[-1], m[:, 0], m[:, -1]])] = True
    ok = (A >= 12) & ~bmask[lab]
    if pore_like is not None: ok &= ~pore_like
    A, cy, cx, myy, mxx, mxy = A[ok], cy[ok], cx[ok], myy[ok], mxx[ok], mxy[ok]
    if len(A) < 15: return {k: np.nan for k in KEYS}
    det = np.sqrt(np.maximum(mxx * myy - mxy ** 2, 1e-9)); Nxx, Nyy, Nxy = mxx / det, myy / det, mxy / det
    out = {}
    w = A / A.sum()
    out['df_glob_lar'] = float(_tensor_lar((w * Nxx).sum(), (w * Nyy).sum(), (w * Nxy).sum()))
    glar = _tensor_lar(Nxx, Nyy, Nxy)
    out['df_grain_lar_wmean'] = float((w * glar).sum()); o = np.argsort(glar); c = np.cumsum(w[o]); out['df_grain_lar_q75'] = float(glar[o][np.searchsorted(c, 0.75)])
    for win in (64, 32):
        vals = []; dirs = []
        for a in range(0, H, win):
            for b in range(0, W, win):
                s = (cy >= a) & (cy < a + win) & (cx >= b) & (cx < b + win)
                if s.sum() < 4: continue
                ws = A[s] / A[s].sum(); Sxx, Syy, Sxy = (ws * Nxx[s]).sum(), (ws * Nyy[s]).sum(), (ws * Nxy[s]).sum()
                vals.append(_tensor_lar(Sxx, Syy, Sxy)); dirs.append(np.array([Sxx - Syy, 2 * Sxy]))
        vals = np.array(vals)
        out[f'df_loc{win}_lar_mean'] = float(vals.mean()) if len(vals) else np.nan; out[f'df_loc{win}_lar_std'] = float(vals.std()) if len(vals) > 2 else np.nan
        if win == 64:
            D = np.array(dirs).reshape(-1, 2); nrm = np.linalg.norm(D, axis=1).sum()
            out['df_dir_coh64'] = float(np.linalg.norm(D.sum(0)) / max(nrm, 1e-9)) if len(D) else np.nan
    la = np.log(A); X = np.c_[np.ones(len(A)), (cy - H / 2) / H, (cx - W / 2) / W]
    c1, *_ = np.linalg.lstsq(X, la, rcond=None); r1 = la - X @ c1
    out['sg_slope'] = float(np.hypot(c1[1], c1[2])); out['sg_r2'] = float(1 - r1.var() / max(la.var(), 1e-9))
    X2 = np.c_[X, X[:, 1] ** 2, X[:, 2] ** 2, X[:, 1] * X[:, 2]]; c2, *_ = np.linalg.lstsq(X2, la, rcond=None); r2 = la - X2 @ c2
    out['sg_quad_r2'] = float(1 - r2.var() / max(la.var(), 1e-9))
    tv = []
    for a in range(0, H, 64):
        for b in range(0, W, 64):
            s = (cy >= a) & (cy < a + 64) & (cx >= b) & (cx < b + 64)
            if s.sum() >= 3: tv.append(np.log(A[s].mean()))
    out['sg_loc64_std'] = float(np.std(tv)) if len(tv) > 4 else np.nan
    out['df_lar_vs_size_corr'] = float(np.corrcoef(glar, la)[0, 1])
    out['df_stretch_x_size'] = float(out['df_glob_lar'] * np.exp(la.mean()))
    return out

def matrix(masks):
    rows = [features(m) for m in masks]
    return np.array([[r[k] for k in KEYS] for r in rows], float)
