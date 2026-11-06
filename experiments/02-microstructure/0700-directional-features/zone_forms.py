"""Aligned 4x4 zones (64 px): zone-level log grain size from U-Net maps; which aggregate forms carry hardness signal beyond het4_u."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
srcs = ['../push-400/unet2maps-m1-train.npz', '../push-400/unet2maps-m2-train.npz', '../push-400/unet2maps-h1-train.npz', '../push-400/unet4maps-z1-train.npz']
ids = None; Z = None
for f in srcs:
    M = np.load(_repo_io(f), allow_pickle=True); ii = [str(i) for i in M['ids']]; ids = ii if ids is None else ids; assert ii == ids
    zz = np.zeros((500, 16, 3))
    for n in range(500):
        m = M['maps'][n].astype(np.float64); g = m[0] + m[1]; la = m[4]; a = np.exp(la)
        Wz = g.reshape(4, 32, 4, 32).sum((1, 3)).ravel()
        zz[n, :, 0] = (g * la).reshape(4, 32, 4, 32).sum((1, 3)).ravel() / Wz                       # pixel-weighted mean log area
        zz[n, :, 1] = -2 * np.log((g * np.exp(-0.5 * la)).reshape(4, 32, 4, 32).sum((1, 3)).ravel() / Wz)   # Hall-Petch weighted
        zz[n, :, 2] = np.log(Wz * 4 / (g / a).reshape(4, 32, 4, 32).sum((1, 3)).ravel())                # number-weighted (area / count)
    Z = zz if Z is None else Z + zz
Z /= len(srcs); np.save(_repo_io('zones16-train.npy'), Z)
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X']); assert [str(i) for i in F['ids']] == ids
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
nz = X[:, keys.index('noise')] * 255
for q, qn in enumerate(['la', 'hp', 'num']):
    v = Z[:, :, q]; s_ = np.sort(v, 1)
    forms = {'std': v.std(1), 'range': s_[:, -1] - s_[:, 0], 'max-mean': s_[:, -1] - v.mean(1), 'mean-min': v.mean(1) - s_[:, 0], 'iqr': s_[:, 12] - s_[:, 3],
             'mad': np.mean(np.abs(v - v.mean(1, keepdims=True)), 1), 'std_exp': np.exp(v).std(1) / np.exp(v).mean(1), 'mean_hp': np.mean(np.exp(-0.5 * v), 1), 'mean': v.mean(1)}
    for nm, s in [('HQ', nz < 11.9), ('all', nz > -1)]:
        for cn, ctrl in [('base', ['u_ph', 'z_f25', 'z_f50', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac']), ('+het4_u', ['u_ph', 'z_f25', 'z_f50', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac', 'het4_u', 'het4_u_mx'])]:
            C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]], A[s, :2]]; ry = y[s] - C @ np.linalg.lstsq(C, y[s], rcond=None)[0]
            print(qn, nm, '%-8s' % cn, ' '.join('%s %+.3f' % (k, np.corrcoef(f[s] - C @ np.linalg.lstsq(C, f[s], rcond=None)[0], ry)[0, 1]) for k, f in forms.items()))
