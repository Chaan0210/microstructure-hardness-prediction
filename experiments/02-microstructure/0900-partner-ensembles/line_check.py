"""Label-free (train only): are grain-boundary lines partly missing in noisy-preset images?
Per boundary pixel (4-neighbour label change, both sides non-zero labels, both grains >= 30 px): dip = (mean of the two grains' eroded interior means
- pixel value) / S, S = image bright level (75th pct of grain means). Clean rows: masks-v3; noisy rows (noise >= 12): ft-H300 1-view masks.
Simulated: clean rows degraded with our calibrated extra blur (sigma ~0.75, calib_own.py) + noise drawn from the real noisy levels, lines kept (a) or each grain side
removed with p = 0.5 (b); dips measured with the clean masks. Pixel dips are smoothed by the sigma-1 blur in all cases."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]
Mc = np.load(_repo_path(E, 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); ids = [str(i) for i in Mc['ids']]
Mn = np.load(_repo_path(E, 'push-300/ftseg-ft-H300-train-noisy-n1-masks.npz'), allow_pickle=True); npos = {str(i): k for k, i in enumerate(Mn['ids'])}
nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255; noisy_levels = nz[nz >= 12]

def dips(img, m):
    m = m.astype(int); sm = ndi.gaussian_filter(img, 1.0)
    er = m * (ndi.minimum_filter(m, 5) == ndi.maximum_filter(m, 5)); L = np.arange(m.max() + 1)
    cnt = np.bincount(er.ravel(), minlength=len(L)); mu = np.bincount(er.ravel(), weights=sm.ravel(), minlength=len(L)) / np.maximum(cnt, 1)
    ok = cnt >= 30; ok[0] = False; S = np.quantile(mu[ok], 0.75) if ok.sum() > 5 else np.nan
    out = []
    for a, b, v in ((m[:, :-1], m[:, 1:], (sm[:, :-1] + sm[:, 1:]) / 2), (m[:-1, :], m[1:, :], (sm[:-1, :] + sm[1:, :]) / 2)):
        s = (a != b) & ok[a] & ok[b]; out.append(((mu[a[s]] + mu[b[s]]) / 2 - v[s]) / S)
    return np.concatenate(out)

def degrade(a, m, r, noise, remove_p):
    m = m.astype(int); f = a.copy()
    if remove_p > 0:
        er = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3)); cnt = np.bincount(er.ravel(), minlength=m.max() + 1)
        mu = np.bincount(er.ravel(), weights=a.ravel(), minlength=m.max() + 1) / np.maximum(cnt, 1)
        near = ndi.grey_dilation(m, size=3) if (m == 0).any() else m
        lab = np.where(m > 0, m, near); bnd = (m == 0) | (ndi.minimum_filter(m, 3) != ndi.maximum_filter(m, 3))
        drop = r.rand(m.max() + 1) < remove_p; sel = bnd & drop[lab] & (cnt[lab] > 0)
        f[sel] = mu[lab[sel]] + r.normal(0, 2.0, sel.sum())
    f = ndi.gaussian_filter(f, abs(r.normal(0.75, 0.15)))   # extra blur fitted on our train images (calib_own.py)
    return np.clip(np.round(f + r.normal(0, noise, f.shape)), 0, 255)

r = np.random.RandomState(0); D = {'clean': [], 'noisy_real': [], 'sim_keep': [], 'sim_drop50': []}
for n, i in enumerate(ids):
    img = np.array(Image.open(_repo_path(P, f'data/train/{i}.png'))).astype(np.float64)
    if nz[n] < 9.5:
        m = Mc['masks'][n]; D['clean'].append(np.median(dips(img, m)))
        nl = float(r.choice(noisy_levels))
        D['sim_keep'].append(np.median(dips(degrade(img, m, r, nl, 0.0), m)))
        D['sim_drop50'].append(np.median(dips(degrade(img, m, r, nl, 0.5), m)))
    elif nz[n] >= 12 and i in npos:
        D['noisy_real'].append(np.median(dips(img, Mn['masks'][npos[i]])))
for k, v in D.items():
    v = np.array(v); v = v[np.isfinite(v)]; print(f'{k:11s} n={len(v):3d}  median boundary dip / S  {np.median(v):.4f}  IQR {np.percentile(v, 25):.4f}..{np.percentile(v, 75):.4f}')
