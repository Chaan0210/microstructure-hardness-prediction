"""Probe: phase fraction after anisotropic smoothing along the dominant elongation direction (label-free)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
from PIL import Image
from scipy import ndimage as ndi
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X']); ids = [str(i) for i in F['ids']]
A = np.load(_repo_io('angle-train.npz'))['X']
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
def aniso(im, th, sa, sc):
    # rotate so the elongation direction is horizontal, smooth (sc across rows, sa along cols), rotate back
    deg = np.degrees(th)
    r = ndi.rotate(im, -deg, reshape=True, order=1, mode='reflect')
    r = ndi.gaussian_filter(r, (sc, sa))
    b = ndi.rotate(r, deg, reshape=True, order=1, mode='reflect')
    oy = (b.shape[0] - im.shape[0]) // 2; ox = (b.shape[1] - im.shape[1]) // 2
    return b[oy:oy + im.shape[0], ox:ox + im.shape[1]]
out = np.full((500, 6), np.nan)
for n in range(500):
    m = maps[n].astype(np.float32); c2, s2 = m[5], m[6]; g = m[0] + m[1]
    C = (g * c2).sum() / g.sum(); S = (g * s2).sum() / g.sum(); th = 0.5 * np.arctan2(S, C)   # doubled-angle -> direction (map coords)
    im = np.array(Image.open(_repo_io(f'../../data/train/{ids[n]}.png'))).astype(np.float32)
    pore = np.kron((m[2] > 0.5).astype(np.float32), np.ones((2, 2))) > 0.5
    row = []
    for sa, sc in [(6, 0.7), (10, 0.7)]:
        for sgn in (1, -1):
            f = aniso(im, sgn * th, sa, sc)
            vals = f[~pore]; mat = np.percentile(vals, 75); lo = np.percentile(f, 0.5)
            z = (mat - f) / max(mat - lo, 1e-3)
            row.append(((z > 0.25) & ~pore).sum() / (~pore).sum())
    iso = ndi.gaussian_filter(im, 2.0); vals = iso[~pore]; mat = np.percentile(vals, 75); lo = np.percentile(iso, 0.5); z = (mat - iso) / max(mat - lo, 1e-3)
    row.append(((z > 0.25) & ~pore).sum() / (~pore).sum())
    row.append(th)
    out[n] = row
np.save(_repo_io('dirsmooth-train.npy'), out)
nz = X[:, keys.index('noise')] * 255; el = X[:, keys.index('m_el_glob')]
ctrl = ['het4_u', 'sizefield_std_s32', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj']
names = ['an6_+', 'an6_-', 'an10_+', 'an10_-', 'iso2']
for nm, s in [('HQ el>0.6', (nz < 11.9) & (el > 0.6)), ('LQ el>0.6', (nz >= 11.9) & (el > 0.6)), ('LQ el<0.5', (nz >= 11.9) & (el < 0.5))]:
    C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
    line = []
    for j, k in enumerate(names):
        v = out[s, j]; rv = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]; line.append('%s %+.2f' % (k, np.corrcoef(rv, ry)[0, 1]))
    v = X[s, keys.index('u_ph_hard')]; rv = v - C @ np.linalg.lstsq(C, v, rcond=None)[0]; line.append('u_ph_hard %+.2f' % np.corrcoef(rv, ry)[0, 1])
    # incremental over u_ph_hard + z_f25
    C2 = np.c_[C, X[s][:, [keys.index('u_ph_hard'), keys.index('z_f25'), keys.index('z_f50')]]]; ry2 = y[s] - C2 @ np.linalg.lstsq(C2, y[s], rcond=None)[0]
    line.append('| beyond U-Net: ' + ' '.join('%s %+.2f' % (k, np.corrcoef(out[s, j] - C2 @ np.linalg.lstsq(C2, out[s, j], rcond=None)[0], ry2)[0, 1]) for j, k in enumerate(names)))
    print(nm, s.sum(), ' '.join(line))
