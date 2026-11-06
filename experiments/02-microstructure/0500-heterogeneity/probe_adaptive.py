"""Clean half, exact masks: per-image adaptive phase split (2-component area-weighted mixture on grain darkness z) vs fixed z thresholds."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
sys.path.insert(0, '.')
import resid035 as R
lab = R.labels(); nz = R.noise()
F = np.load(_repo_io('feat-u1dmavgH2Z-train.npz'), allow_pickle=True); k = [str(x) for x in F['keys']]; ids = [str(i) for i in F['ids']]
y = np.array([lab[i] for i in ids]); n = np.array([nz[i] for i in ids]) * 255
ctrl = ['m_la_mean', 'm_la_f32', 'm_la_tile8', 'm_el_glob', 'm_el_pix', 'stcoh_s3_w8_locmed', 'stcoh_s1.5_w32_glob', 'pore_frac', 'm_la_std', 'het4_u']
Xc = F['X'][:, [k.index(c) for c in ctrl]].astype(float); Xc = np.where(np.isnan(Xc), np.nanmedian(Xc, 0), Xc); C = np.c_[np.ones(len(y)), Xc]
uu = np.c_[F['X'][:, k.index('m_ph_hard')], F['X'][:, k.index('u_ph_hard')], F['X'][:, k.index('z_f25')], F['X'][:, k.index('z_f50')]]
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mid = {str(i): j for j, i in enumerate(M['ids'])}
sel = np.where(n < np.median(n))[0]; V = {}; keep = []
for j in sel:
    i = ids[j]; a = np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(float); m = M['masks'][mid[i]].astype(np.int64)
    sm = ndi.gaussian_filter(a / 255, 1.0); inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}; ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = [l for l in props if l in ipr and ipr[l].area >= 3]
    if len(labs) < 8: continue
    A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1) for l in labs]); SOL = np.array([props[l].solidity for l in labs])
    o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3 * np.median(A)); g = ~pore; Ag, zg = A[g], z[g]
    # 2-component mixture on z (area weighted), init split at 0.25
    ph = zg > 0.25
    if ph.sum() < 2 or (~ph).sum() < 2: continue
    mu = np.array([np.average(zg[~ph], weights=Ag[~ph]), np.average(zg[ph], weights=Ag[ph])]); sd = np.array([0.05, 0.08]); pi = np.array([0.8, 0.2])
    for _ in range(50):
        L = np.stack([pi[q] * np.exp(-0.5 * ((zg - mu[q]) / sd[q]) ** 2) / sd[q] for q in range(2)]) + 1e-12; Rr = L / L.sum(0); w = Rr * Ag
        pi = w.sum(1) / w.sum(); mu = (w * zg).sum(1) / w.sum(1); sd = np.sqrt((w * (zg - mu[:, None]) ** 2).sum(1) / w.sum(1)) + 0.01
    hiC = int(np.argmax(mu))
    V[j] = dict(gmm_soft=(Rr[hiC] * Ag).sum() / Ag.sum(), gmm_hard=Ag[Rr[hiC] > 0.5].sum() / Ag.sum(), gmm_mu_ph=mu[hiC], gmm_mu_mx=mu[1 - hiC], gmm_gap=(mu[hiC] - mu[1 - hiC]) / np.sqrt(sd[0] * sd[1]),
                f25=Ag[zg > 0.25].sum() / Ag.sum(), mid_thr=Ag[zg > 0.5 * (mu[0] + mu[1])].sum() / Ag.sum())
    keep.append(j)
keep = np.array(keep)
def pc(v, ex=None):
    A_ = C[keep] if ex is None else np.c_[C[keep], ex[keep]]
    ry = y[keep] - A_ @ np.linalg.lstsq(A_, y[keep], rcond=None)[0]; rv = v - A_ @ np.linalg.lstsq(A_, v, rcond=None)[0]; return np.corrcoef(rv, ry)[0, 1]
print('n', len(keep))
for kk in V[keep[0]]:
    v = np.array([V[j][kk] for j in keep]); print(f'{kk:10s} raw {np.corrcoef(v, y[keep])[0, 1]:+.3f} pcorr {pc(v):+.3f} beyond U-Net phase+z {pc(v, uu):+.3f}  mean {v.mean():.3f} sd {v.std():.3f}')
