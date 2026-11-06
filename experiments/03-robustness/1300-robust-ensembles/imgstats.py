"""Pixel-level domain-gap statistics: real noisy train images (noise >= 12) vs degraded copies (deg4) vs clean train images.
radial PSD (normalised by total), noise estimate, block-mean (8 px) spread after median filtering, gradient energy after Gaussian 1.5."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
from PIL import Image
from scipy import ndimage as ndi
P = '../..'; F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]
ids = [str(i) for i in F['ids']]; nz = np.nan_to_num(F['X'][:, keys.index('noise')].astype(float) * 255, nan=0)
yy, xx = np.mgrid[-128:128, -128:128]; rr = np.hypot(yy, xx).astype(int)
def stats(path):
    a = np.asarray(Image.open(_repo_io(path)).convert('L'), float); a0 = a - a.mean()
    ps = np.fft.fftshift(np.abs(np.fft.fft2(a0)) ** 2); rad = np.bincount(rr.ravel(), ps.ravel(), minlength=182)[:128] / np.bincount(rr.ravel(), minlength=182)[:128]
    hf = rad[100:128].mean(); lf = rad[2:16].mean(); mf = rad[30:60].mean()
    sm = ndi.median_filter(a, 5); bm = sm.reshape(32, 8, 32, 8).mean((1, 3))
    q = np.quantile(bm, [0.1, 0.5, 0.9]); gr = np.hypot(*np.gradient(ndi.gaussian_filter(a, 1.5)))
    return [np.log(hf), np.log(mf / hf), np.log(lf / hf), q[1] - q[0], q[2] - q[1], np.median(gr), np.quantile(gr, 0.95), a.std()]
names = ['log hf power', 'log mid/hf', 'log low/hf', 'blockmean p50-p10', 'blockmean p90-p50', 'grad med', 'grad p95', 'img sd']
R = np.array([stats(f'{P}/data/train/{ids[i]}.png') for i in np.where(nz >= 12)[0]])
Cn = np.array([stats(f'{P}/data/train/{ids[i]}.png') for i in np.where(nz < 7)[0][:120]])
import glob, json
par = json.load(open(_repo_io(f'{P}/experiments/push-400/synth/train_deg4_params.json')))
S = np.array([stats(f) for f in sorted(glob.glob(f'{P}/experiments/push-400/synth/train_deg4/*.png'))[:160]])
for j, n in enumerate(names): print(f'{n:20s} real-noisy {R[:, j].mean():8.3f}±{R[:, j].std():.3f}   copy(deg4) {S[:, j].mean():8.3f}±{S[:, j].std():.3f}   clean {Cn[:, j].mean():8.3f}±{Cn[:, j].std():.3f}')
