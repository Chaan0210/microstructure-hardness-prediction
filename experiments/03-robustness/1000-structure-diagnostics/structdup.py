"""Structural near-duplicate search (label-free): are there images that show the same microstructure under a different preset / orientation / shift?
Each image: Gaussian blur (sigma 2) -> 64x64 block mean -> z-score. Pairwise max over the 8 D4 transforms of the zero-shift correlation, plus the
max FFT cross-correlation over circular shifts (catches shifted crops). Reports the top pairs train-train and test-train.
usage: structdup.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
P0 = Path(_repo_io('../..')); tr = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data/train.csv')))]; te = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data/sample_submission.csv')))]
def prep(p):
    a = np.asarray(Image.open(_repo_io(p)), np.float32); b = ndi.gaussian_filter(a, 2.0).reshape(64, 4, 64, 4).mean((1, 3)); return (b - b.mean()) / (b.std() + 1e-6)
X = np.array([prep(_repo_path(P0, 'data/train', f'{i}.png')) for i in tr] + [prep(_repo_path(P0, 'data/test', f'{i}.png')) for i in te]); ids = tr + te; n = len(ids)
def d4(x, k):
    if k & 1: x = np.swapaxes(x, -1, -2)
    if k & 2: x = x[..., ::-1, :]
    if k & 4: x = x[..., :, ::-1]
    return x
F = X.reshape(n, -1); best = np.full((n, n), -1.0)
for k in range(8):
    Fk = d4(X, k).reshape(n, -1); C = F @ Fk.T / F.shape[1]; best = np.maximum(best, C)
np.fill_diagonal(best, -1)
# shift-tolerant: FFT cross-correlation max (circular), on the top-50 candidates per image only
FX = np.fft.rfft2(X); top = np.argsort(-best, 1)[:, :50]; sbest = np.full(n, -1.0); sarg = np.zeros(n, int)
for i in range(n):
    for k in range(8):
        Y = d4(X[top[i]], k); FY = np.fft.rfft2(Y); cc = np.fft.irfft2(FX[i][None] * np.conj(FY), s=(64, 64)).reshape(len(top[i]), -1).max(1) / 4096
        j = int(np.argmax(cc))
        if cc[j] > sbest[i]: sbest[i], sarg[i] = cc[j], top[i][j]
ntr = len(tr)
print('zero-shift D4 max corr: train-train top', np.sort(best[:ntr, :ntr].ravel())[::-1][:10:2].round(3), ' test-train top', np.sort(best[ntr:, :ntr].ravel())[::-1][:5].round(3))
print('distribution of each image best match (zero-shift):', np.quantile(best.max(1), [.5, .9, .99, 1]).round(3))
print('shift-tolerant best per image: quantiles', np.quantile(sbest, [.5, .9, .99, 1]).round(3))
o = np.argsort(-sbest)[:15]
for i in o: print(f'{ids[i]} <-> {ids[sarg[i]]}  shift-corr {sbest[i]:.3f}  zero-shift {best[i, sarg[i]]:.3f}')
np.savez(_repo_io('structdup.npz'), ids=np.array(ids), best=best.astype(np.float32), sbest=sbest, sarg=sarg)
