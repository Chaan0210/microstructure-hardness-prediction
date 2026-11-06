"""Generic orientation-sensitive descriptors (label-free): log power spectrum in radial x angular sectors of the image and of U-Net phase/log-area maps."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
from PIL import Image
ids = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/train.csv')))]
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; assert [str(i) for i in M['ids']] == ids
def spec(f, nr=6, na=8):
    f = f - f.mean(); P = np.abs(np.fft.fftshift(np.fft.fft2(f * np.hanning(f.shape[0])[:, None] * np.hanning(f.shape[1])[None, :]))) ** 2
    h, w = f.shape; yy, xx = np.mgrid[0:h, 0:w]; yy = yy - h // 2; xx = xx - w // 2; r = np.hypot(yy, xx); th = np.mod(np.arctan2(yy, xx), np.pi)
    rb = np.logspace(np.log10(2), np.log10(h // 2), nr + 1); out = []
    for i in range(nr):
        for j in range(na):
            s = (r >= rb[i]) & (r < rb[i + 1]) & (th >= j * np.pi / na) & (th < (j + 1) * np.pi / na); out.append(np.log(P[s].mean() + 1e-9))
    out = np.array(out).reshape(nr, na); return np.r_[(out - out.mean(1, keepdims=True)).ravel(), out.mean(1)]   # angular anisotropy per radius + radial profile
D = []
for n, i in enumerate(ids):
    im = np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(np.float64)
    m = maps[n].astype(np.float64)
    D.append(np.r_[spec(im), spec(m[1]), spec(m[4] * (m[0] + m[1]))])
D = np.array(D); np.save(_repo_io('generic-train.npy'), D); print(D.shape)
