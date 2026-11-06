"""Within-matrix intensity autocorrelation (lag >= 3) as a boundary-free grain-size measurement (label-free, per image).
I -> minus large-scale background (Gaussian 16) -> matrix mask (smoothed intensity above the image's Otsu split of the smoothed image, eroded 3 px,
excludes dark phase and pores) -> masked FFT autocovariance, radial mean at lags 1..40 -> features: C(r)/C(3) at r=6,10,16,24 and the 1/e length.
White noise only enters lag 0, blur lags 1-2. usage: wacf.py train|test -> wacf-<split>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
P = Path(_repo_io('../..')); sp = sys.argv[1]
ids = sorted(f.stem for f in (_repo_path(P, f'experiments/push-400/synth/train_{sp}')).glob('*.png')) if sp.startswith('deg') else [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv' if sp == 'train' else 'data/sample_submission.csv')))]
L = 41; yy, xx = np.mgrid[-L:L + 1, -L:L + 1]; rr = np.rint(np.hypot(yy, xx)).astype(int)
def feats(im):
    I = im.astype(np.float64); R = I - ndi.gaussian_filter(I, 16); S = ndi.gaussian_filter(I, 2.5)
    M = ndi.binary_erosion(S > threshold_otsu(S), iterations=3).astype(np.float64)
    if M.mean() < 0.2: M = ndi.binary_erosion(S > np.quantile(S, 0.3), iterations=3).astype(np.float64)
    R = (R - (R * M).sum() / M.sum()) * M; h, w = I.shape; F = lambda a: np.fft.rfft2(a, s=(2 * h, 2 * w))
    num = np.fft.irfft2(np.abs(F(R)) ** 2, s=(2 * h, 2 * w)); den = np.fft.irfft2(np.abs(F(M)) ** 2, s=(2 * h, 2 * w))
    num = np.fft.fftshift(num)[h - L:h + L + 1, w - L:w + L + 1]; den = np.fft.fftshift(den)[h - L:h + L + 1, w - L:w + L + 1]
    C = num / np.maximum(den, 1); prof = np.array([C[rr == r].mean() for r in range(L)])
    return prof, M.mean()
out, cov = [], []
for i in ids:
    p, c = feats(np.asarray(Image.open(_repo_io((_repo_path(P, f'experiments/push-400/synth/train_{sp}/{i}.png')) if sp.startswith('deg') else (_repo_path(P, f'data/{sp}/{i}.png')))).convert('L'))); out.append(p); cov.append(c)
np.savez(_repo_io(f'wacf-{sp}.npz'), ids=np.array(ids), prof=np.array(out), cov=np.array(cov)); print('done', sp, len(ids))
