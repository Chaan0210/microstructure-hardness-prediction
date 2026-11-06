"""Label-free per-image degradation estimates (blur / contrast / low-frequency shading) from the power spectrum and intensity range.
usage: degest.py SPLIT -> degest-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
H = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]; split = sys.argv[1]
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data', 'train.csv' if split == 'train' else 'sample_submission.csv')))]
yy, xx = np.mgrid[-128:128, -128:128]; rr = np.sqrt(yy ** 2 + xx ** 2) / 256.0; win = np.outer(np.hanning(256), np.hanning(256))
def band(F, lo, hi): m = (rr >= lo) & (rr < hi); return F[m].mean()
KEYS = ['d_noisefloor', 'd_hf_ratio', 'd_mf_ratio', 'd_lowfreq', 'd_range', 'd_shade']
X = []
for i in ids:
    a = np.array(Image.open(_repo_path(P, f'data/{split}/{i}.png'))).astype(float); F = np.abs(np.fft.fftshift(np.fft.fft2((a - a.mean()) * win))) ** 2
    nf = band(F, 0.42, 0.5); s1 = band(F, 0.03, 0.06) - nf; s2 = band(F, 0.10, 0.16) - nf; s3 = band(F, 0.16, 0.24) - nf; s0 = band(F, 0.004, 0.012)
    sm = ndi.gaussian_filter(a, 24)[32:-32, 32:-32]
    X.append([np.log(nf), np.log(max(s3, 1) / max(s1, 1)), np.log(max(s2, 1) / max(s1, 1)), np.log(s0 / max(s1, 1)), np.percentile(a, 95) - np.percentile(a, 5), sm.std()])
np.savez(_repo_path(H, f'degest-{split}.npz'), ids=np.array(ids), X=np.array(X), keys=np.array(KEYS)); print(split, np.array(X).shape)
