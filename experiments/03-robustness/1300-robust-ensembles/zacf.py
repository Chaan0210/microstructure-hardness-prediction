"""Segmentation-free zone grain-size field from the image autocovariance: for each 64x64 zone of the 4x4 grid (the generator's het grid),
linear autocovariance (zero-padded FFT, pair-count normalised), lags 0-1 dropped (white noise / blur), normalised by the mean at radius 2;
zone size = integral range over 2 <= |r| <= 16 (pixels^2, ~ grain area). Features: acf_logA_mean, acf_het (std of zone log-area), acf_het_mx.
usage: zacf.py train|test|synth:<set>"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, csv
from PIL import Image
from pathlib import Path
P = Path(_repo_io('../..')); which = sys.argv[1]
if which in ('train', 'test'):
    ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv' if which == 'train' else 'data/sample_submission.csv')))]; files = [_repo_path(P, f'data/{which}/{i}.png') for i in ids]
else:
    s = which.split(':')[1]; files = sorted((_repo_path(P, f'experiments/push-400/synth/train_{s}')).glob('*.png')); ids = [f.stem for f in files]
Z = 64; yy, xx = np.mgrid[-Z + 1:Z, -Z + 1:Z]; rr = np.hypot(yy, xx)
ring2 = (rr >= 1.5) & (rr < 2.5); disk = (rr >= 1.5) & (rr <= 16)
ones = np.ones((Z, Z)); cnt = np.real(np.fft.ifft2(np.abs(np.fft.fft2(ones, (2 * Z, 2 * Z))) ** 2)); cnt = np.fft.fftshift(cnt)[1:, 1:]
def zone_acf(a):
    a = a - a.mean(); F = np.fft.fft2(a, (2 * Z, 2 * Z)); c = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2)))[1:, 1:] / np.maximum(cnt, 1)
    c2 = c[ring2].mean(); return (c[disk] / c2).sum() if c2 > 0 else np.nan, c2, c[Z - 1, Z - 1]
out = []
for f in files:
    im = np.asarray(Image.open(_repo_io(f)).convert('L'), float)
    la = []; amp = []
    for i in range(4):
        for j in range(4):
            A, c2, c0 = zone_acf(im[i * Z:(i + 1) * Z, j * Z:(j + 1) * Z]); la.append(np.log(max(A, 1e-3))); amp.append(c2)
    la = np.array(la); out.append([la.mean(), la.std(), la.max() - la.min(), np.log(np.mean(amp))])
np.savez(_repo_io(f'zacf-{which.replace(":", "_")}.npz'), ids=np.array(ids), X=np.array(out), keys=np.array(['acf_logA_mean', 'acf_het', 'acf_het_mx', 'acf_logamp2']))
print(which, len(ids), np.nanmean(out, 0))
