"""Label-free image-quality statistics for every train/test image (diagnostic): pixel noise (MAD of 2x2 Haar diagonal), relative power in
spatial-frequency bands of the radially averaged spectrum (blur proxy), global contrast (p95-p5). -> preset-stats.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np
from PIL import Image
P = '../../data'
def stats(a):
    a = a.astype(np.float64); hh = (a[::2, ::2] - a[::2, 1::2] - a[1::2, ::2] + a[1::2, 1::2]) / 2; nz = np.median(np.abs(hh)) / .6745
    F = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean()))) ** 2; yy, xx = np.mgrid[-128:128, -128:128]; r = np.hypot(yy, xx)
    band = lambda lo, hi: F[(r >= lo) & (r < hi)].mean()
    white = nz ** 2 * 65536   # expected white-noise power per frequency bin
    return [nz, np.log(band(4, 8)), np.log(band(8, 16)), np.log(max(band(16, 32) - white, 1.0)), np.log(max(band(32, 64) - white, 1.0)), np.percentile(a, 95) - np.percentile(a, 5)]
out = {}
for split, f in (('train', 'train.csv'), ('test', 'sample_submission.csv')):
    ids = [r['ID'] for r in csv.DictReader(open(_repo_io(f'{P}/{f}')))]
    out[split] = np.array([stats(np.asarray(Image.open(_repo_io(f'{P}/{split}/{i}.png')))) for i in ids]); out[split + '_ids'] = np.array(ids)
np.savez(_repo_io('preset-stats.npz'), **out, keys=np.array(['noise', 'lp4_8', 'lp8_16', 'lp16_32_ex', 'lp32_64_ex', 'contrast']))
for split in ('train', 'test'):
    X = out[split]; m = X[:, 0] >= 12
    print(split, 'noisy n', m.sum(), ' corr(noise, lp16_32_ex) %.2f' % np.corrcoef(X[m, 0], X[m, 3])[0, 1])
