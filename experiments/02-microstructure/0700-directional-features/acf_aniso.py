"""Global structural anisotropy from the 2D autocorrelation of the U-Net boundary probability map (z1, 128 res): second-moment tensor of the
central ACF lobe -> eigenvalue ratio (log stretch estimate) and its 4-fold angle terms. Label-free."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv, sys
split = sys.argv[1]
M = np.load(_repo_io(f'../push-400/unet4maps-z1-{split}.npz'), allow_pickle=True); ids = [str(i) for i in M['ids']]; maps = M['maps']
yy, xx = np.mgrid[-64:64, -64:64]; out = np.zeros((len(ids), 4))
for n in range(len(ids)):
    b = maps[n, 3].astype(np.float64); b = b - b.mean()
    P = np.abs(np.fft.fft2(b)) ** 2; acf = np.fft.fftshift(np.real(np.fft.ifft2(P))); acf /= acf.max()
    for thr in (0.2,):
        w = np.where(acf > thr, acf, 0.0); lab_ok = (np.abs(yy) < 40) & (np.abs(xx) < 40); w = w * lab_ok
        Ixx = (w * xx * xx).sum() / w.sum(); Iyy = (w * yy * yy).sum() / w.sum(); Ixy = (w * xx * yy).sum() / w.sum()
        ev = np.linalg.eigvalsh(np.array([[Ixx, Ixy], [Ixy, Iyy]])); lr = 0.5 * np.log(ev[1] / max(ev[0], 1e-9))
        th = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
        out[n] = [lr, np.log(np.sqrt(ev[0] * ev[1])), lr * np.cos(4 * th), lr * np.sin(4 * th)]
np.savez(_repo_io(f'acf-{split}.npz'), ids=np.array(ids), X=out, keys=np.array(['acf_logratio', 'acf_logsize', 'acf_c4', 'acf_s4']))
print('done', split, out.mean(0).round(3), out.std(0).round(3))
