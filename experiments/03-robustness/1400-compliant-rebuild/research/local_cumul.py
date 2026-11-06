"""Block-level label-blind cumulant phase measurements (H9 measure() per 3x3 block of the 216-px core): spatial heterogeneity of the dark-phase
fraction / contrast that is insensitive to additive Gaussian noise (k3, k4 unaffected; m2 noise-corrected). Inputs: raw image, U-Net z1 maps, noise.
usage: local_cumul.py ZMAPDIR FEATDIR OUT  (ZMAPDIR/unet4maps-z1-{split}.npz, FEATDIR/feat-u1dmavgH2Z-anghcor-{split}.npz for ids/noise)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
ZD, FD, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]; P = Path(str(_REPO_ROOT))
PGRID = np.linspace(.005, .495, 2001); U = PGRID * (1 - PGRID); RATIO = (1 - 6 * U) / (U ** (1 / 3) * (1 - 4 * U) ** (2 / 3))
def stats(t, w, noisevar):
    w = w / max(w.sum(), 1e-8); t = t - np.sum(w * t); m2 = float(np.sum(w * t * t)); k3 = float(np.sum(w * t ** 3)); k4 = float(np.sum(w * t ** 4) - 3 * m2 * m2)
    sv = max(m2 - noisevar, 1e-6); ratio = k4 / max(abs(k3), 1e-9) ** (4 / 3); small = float(np.interp(ratio, RATIO[::-1], PGRID[::-1])); p34 = small if k3 < 0 else 1 - small
    contrast = (abs(k3) / max(p34 * (1 - p34) * abs(1 - 2 * p34), 1e-8)) ** (1 / 3); return p34, contrast, np.sqrt(sv)
def measure(x, maps, noise):
    p0 = np.repeat(np.repeat(maps[0].astype(float), 2, 0), 2, 1); grain = np.repeat(np.repeat((maps[0] + maps[1]).astype(float), 2, 0), 2, 1)
    sw = np.clip(p0, 0, 1) ** 4; gw = np.clip(grain, 0, 1) ** 2
    shade = ndi.gaussian_filter(x * sw, 48) / np.maximum(ndi.gaussian_filter(sw, 48), 1e-6); core = (slice(20, -20), slice(20, -20))
    ref = max(float(np.sum(gw[core] / gw[core].sum() * shade[core])), 1.)
    out = []
    for sigma in (1., 2.):
        r = int(4 * sigma + .5); ax = np.arange(-r, r + 1); k = np.exp(-.5 * (ax / sigma) ** 2); k /= k.sum(); nv = noise ** 2 * (np.sum(k ** 2) ** 2)
        sig = (ndi.gaussian_filter(x, sigma) - shade)[core]; w = gw[core]; P3, C3 = [], []
        for i in range(3):
            for j in range(3):
                b = (slice(72 * i, 72 * (i + 1)), slice(72 * j, 72 * (j + 1))); p, c, _ = stats(sig[b], w[b], nv); P3.append(p); C3.append(c / ref)
        P3, C3 = np.array(P3), np.array(C3); out += [P3.std(), P3.max() - P3.min(), C3.std(), np.median(P3)]
    return np.array(out)
KEYS = ['lc1_p_std', 'lc1_p_rng', 'lc1_c_std', 'lc1_p_med', 'lc2_p_std', 'lc2_p_rng', 'lc2_c_std', 'lc2_p_med']
res = {}
for sp in ('train', 'test'):
    F = np.load(_repo_io(FD / f'feat-u1dmavgH2Z-anghcor-{sp}.npz'), allow_pickle=True); k = [str(q) for q in F['keys']]; ids = [str(i) for i in F['ids']]
    noise = np.nan_to_num(F['X'][:, k.index('noise')].astype(float) * 255, nan=0); u = np.load(_repo_io(ZD / f'unet4maps-z1-{sp}.npz')); ix = {str(i): j for j, i in enumerate(u['ids'])}
    res[sp] = np.array([measure(np.asarray(Image.open(_repo_path(P, f'data/{sp}/{i}.png')), float), u['maps'][ix[i]], noise[j]) for j, i in enumerate(ids)]); res[sp + '_ids'] = np.array(ids)
    print(sp, res[sp].shape, flush=True)
np.savez(_repo_io(OUT), keys=np.array(KEYS), **res)
