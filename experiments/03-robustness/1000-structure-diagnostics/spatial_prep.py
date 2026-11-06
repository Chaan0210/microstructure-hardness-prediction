"""Inputs for the two spatial branches (label-free; from the production z1 U-Net maps push-400/unet4maps-z1-<split>.npz, 8 x 128 x 128:
P(matrix), P(phase), P(pore), P(boundary), logA, c2, s2, z darkness).
(a) 2-D correlation maps, lags -16..16 (128-px grid = 2 px per lag), not summarised: autocorrelation of P(phase), P(boundary), z, and the
    phase x pore cross-correlation (FFT, mean removed, normalised by the product of SDs; circular) + the 4 variances -> acf-<split>.npz
(b) patch-token bank in the grain-set format (setZ/run.py): 16 x 16 tokens of 8 x 8 cells, token = cell means of the 8 channels + SD of P(phase)
    and z + (y, x) position; global_X = 12 core structure columns of the 044-slot features -> tokbank-<split>.npz
usage: spatial_prep.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]
CORE = ['u_ph', 'm_ph_hard', 'aw_inv_sqrt_d', 'grain_inv_sqrt_d', 'het4_u', 'pore_frac', 'ar_wmean', 'orient_coherence', 'ang_c4', 'n_obj', 'sh_cor15', 'noise']
L = 16
def xcorr(a, b):
    a = a - a.mean(); b = b - b.mean(); sa, sb = a.std() + 1e-6, b.std() + 1e-6
    c = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape) / a.size / (sa * sb)
    c = np.fft.fftshift(c); m = a.shape[0] // 2; return c[m - L:m + L + 1, m - L:m + L + 1].astype(np.float32)
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P0, 'data/train.csv')))}
for split in ('train', 'test'):
    D = np.load(_repo_path(E, f'push-400/unet4maps-z1-{split}.npz'), allow_pickle=True); ids = [str(i) for i in D['ids']]; M = D['maps']
    F = np.load(_repo_path(E, f'push-800/feat-u1dmavgH2Z-anghcor-{split}.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; fp = {str(i): k for k, i in enumerate(F['ids'])}
    A = np.zeros((len(ids), 4, 2 * L + 1, 2 * L + 1), np.float32); amp = np.zeros((len(ids), 4), np.float32)
    objs, offs = [], [0]; gy, gx = np.mgrid[0:16, 0:16] / 15.0 - 0.5
    for n in range(len(ids)):
        m = M[n].astype(np.float32); ph, po, bd, z = m[1], m[2], m[3], m[7]
        A[n, 0] = xcorr(ph, ph); A[n, 1] = xcorr(bd, bd); A[n, 2] = xcorr(z, z); A[n, 3] = xcorr(ph, po)
        amp[n] = [ph.var(), bd.var(), z.var(), np.mean((ph - ph.mean()) * (po - po.mean()))]
        cell = m.reshape(8, 16, 8, 16, 8).transpose(1, 3, 0, 2, 4)          # [16, 16, 8 ch, 8, 8]
        tok = np.concatenate([cell.mean((3, 4)), cell[:, :, 1].std((2, 3))[..., None], cell[:, :, 7].std((2, 3))[..., None], gy[..., None], gx[..., None]], -1)
        objs.append(tok.reshape(256, -1)); offs.append(offs[-1] + 256)
    np.savez(_repo_path(H, f'acf-{split}.npz'), ids=np.array(ids), A=A, amp=amp)
    G = np.array([[F['X'][fp[i], fk.index(c)] for c in CORE] for i in ids], float)
    okeys = ['p_matrix', 'p_phase', 'p_pore', 'p_boundary', 'logA', 'c2', 's2', 'z', 'sd_phase', 'sd_z', 'pos_y', 'pos_x']
    out = dict(ids=np.array(ids), offsets=np.array(offs), objects=np.concatenate(objs).astype(np.float64), object_keys=np.array(okeys),
               weights=np.full(offs[-1], 1 / 256.0), global_X=G, global_keys=np.array(CORE))
    out['y'] = np.array([lab[i] for i in ids]) if split == 'train' else np.zeros(len(ids))
    np.savez(_repo_path(H, f'tokbank-{split}.npz'), **out); print(split, len(ids), 'acf', A.shape, 'tokens', out['objects'].shape, flush=True)
