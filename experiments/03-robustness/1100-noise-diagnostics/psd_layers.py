"""Is the grain layer of real noisy-preset images blurred more than the pores? (train images only, label-free)
Earlier calibrations (push-300 PSD, push-900 pore-edge erf width) are dominated by the dark, sharp pores. Here pores are masked (dark blobs on a
smoothed image, dilated) and filled with a smooth local estimate, then the radially averaged power spectrum of the grain layer is compared between
noisy (>= 12 gray) and clean (< 8 gray) images in the same grain-size bin. Ratio after removing the white-noise floor -> c^2 exp(-(2 pi f s)^2).
Also the same with pores kept, for the earlier (whole-image) reading."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]
F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); K = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
NZ = np.nan_to_num(F['X'][:, K.index('noise')] * 255, nan=0); NOBJ = F['X'][:, K.index('n_obj')]
def ng(a): hh = (a[::2, ::2] - a[::2, 1::2] - a[1::2, ::2] + a[1::2, 1::2]) / 2; return float(np.median(np.abs(hh)) / .6745)
def depore(a):
    s = ndi.gaussian_filter(a, 2.0); med = np.median(s); mad = np.median(np.abs(s - med)) * 1.4826 + 1e-6
    pm = ndi.binary_dilation(s < med - max(6 * mad, 25), iterations=3)
    if pm.mean() > 0.2: pm[:] = False
    fill = ndi.gaussian_filter(np.where(pm, med, a), 6.0)          # smooth background estimate inside the pore mask
    w = ndi.gaussian_filter((~pm).astype(float), 6.0); fill = ndi.gaussian_filter(np.where(pm, 0, a), 6.0) / np.maximum(w, 1e-3)
    return np.where(pm, fill, a), pm
N = 256; fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing='ij'); FR = np.sqrt(fx ** 2 + fy ** 2)
EDG = np.linspace(0, 0.5, 33); RB = np.digitize(FR, EDG) - 1; WIN = np.outer(np.hanning(N), np.hanning(N))
def rpsd(a):
    a = (a - a.mean()) * WIN; P = np.abs(np.fft.fft2(a)) ** 2 / (WIN ** 2).sum()
    return np.array([P[RB == k].mean() for k in range(32)])
fc = 0.5 * (EDG[:-1] + EDG[1:])
groups = {'clean': np.where(NZ < 8)[0], 'noisy': np.where(NZ >= 12)[0]}
spec = {}
for g, rows in groups.items():
    for mode in ('raw', 'depore'):
        S = []; nn = []
        for r in rows:
            a = np.asarray(Image.open(_repo_path(P0, 'data/train', f'{ids[r]}.png'))).astype(float); nn.append(ng(a))
            b = depore(a)[0] if mode == 'depore' else a; S.append(rpsd(b))
        spec[g, mode] = (np.array(S), np.array(nn))
bins = np.nanpercentile(NOBJ, [0, 33, 67, 100])
def model(f, c2, s): return c2 * np.exp(-(2 * np.pi * f * s) ** 2)
for mode in ('raw', 'depore'):
    Sc, nc = spec['clean', mode]; Sn, nnz = spec['noisy', mode]
    for b in range(3):
        mc = (NOBJ[groups['clean']] >= bins[b]) & (NOBJ[groups['clean']] <= bins[b + 1]); mn = (NOBJ[groups['noisy']] >= bins[b]) & (NOBJ[groups['noisy']] <= bins[b + 1])
        c = Sc[mc].mean(0) - np.mean(nc[mc] ** 2); n = Sn[mn].mean(0) - np.mean(nnz[mn] ** 2)   # white-noise floor = sigma^2 per pixel
        ratio = n / c; sel = (fc > 0.01) & (fc < 0.22) & (c > 0)
        try: p, _ = curve_fit(model, fc[sel], ratio[sel], p0=(0.7, 1.0), maxfev=20000)
        except Exception: p = (np.nan, np.nan)
        print(f'{mode:6s} n_obj bin {b} ({mc.sum()} clean / {mn.sum()} noisy): fit contrast {np.sqrt(abs(p[0])):.3f} extra blur sigma {abs(p[1]):.2f}  ratio@f '
              + ' '.join(f'{f:.2f}:{r:.2f}' for f, r in zip(fc[1:22:3], ratio[1:22:3])), flush=True)
