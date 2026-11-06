"""H3 measurement: global intensity + radially averaged power spectrum (label-free, per image). Writes h3/spec-train.npz.
Columns: mean, sd, and log power in 10 log-spaced radial frequency bands (D4-invariant by construction), plus the
noise-floor-subtracted signal power of the same bands using the image's own high-frequency floor."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[4]
E = _RepoPath(__file__).resolve().parents[3]
fy = np.fft.fftfreq(256)[:, None]; fx = np.fft.fftfreq(256)[None, :]
rad = np.sqrt(fx ** 2 + fy ** 2)
EDGES = np.geomspace(1 / 128, 0.5, 11)
BANDS = [(rad >= EDGES[i]) & (rad < EDGES[i + 1]) for i in range(10)]
HF = rad >= 0.42


def measure(I):
    I = I.astype(float) / 255.0
    m = I.mean(); s = I.std()
    P = np.abs(np.fft.fft2(I - m)) ** 2 / I.size
    bp = np.array([P[b].mean() for b in BANDS])
    floor = P[HF].mean()
    sig = np.maximum(bp - floor, 1e-9)
    return np.r_[m, s, np.log(bp), np.log(sig), np.log(floor)]


KEYS = ['im_mean', 'im_sd'] + [f'logP{i}' for i in range(10)] + [f'logS{i}' for i in range(10)] + ['logfloor']

if __name__ == '__main__':
    inp = np.load(_repo_path(E, 'rmse-research-20261015/inputs.npz'), allow_pickle=True)
    ids = [str(i) for i in inp['ids']]
    X = np.array([measure(np.array(Image.open(_repo_path(ROOT, f'data/train/{i}.png')))) for i in ids])
    np.savez_compressed(_repo_path(HERE, 'spec-train.npz'), ids=np.array(ids), X=X, keys=np.array(KEYS), edges=EDGES)
    print(X.shape, X.mean(0).round(3))
