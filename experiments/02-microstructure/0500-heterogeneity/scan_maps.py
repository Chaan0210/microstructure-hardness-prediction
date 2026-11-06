"""Exploratory residual scan of candidate statistics from multi-task U-Net maps (all rows). Consistency across nets is required."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy.stats import spearmanr
from scipy import ndimage as ndi
sys.path.insert(0, '.')
import resid035 as R
ids, y, B, G, P = R.load(); r = y - B.mean(0); nz = R.noise(); n = np.array([nz[i] for i in ids]) * 255; t = np.digitize(n, np.quantile(n, [1/3, 2/3]))
net = sys.argv[1]
M = dict(np.load(_repo_io(f'../push-400/unet2maps-{net}-train.npz'))); mi = {str(i): k for k, i in enumerate(M['ids'])}
def bs(v, nb): s = 128 // nb; return v.reshape(nb, s, nb, s).sum((1, 3))
def feats(m):
    p0, p1, p2, p3, la, c2, s2 = m; g = p0 + p1; d = {}
    for nb in (2, 4, 8):
        den = bs(g, nb); ok = den > 0.2 * (128 // nb) ** 2
        lg = bs(g * la, nb)[ok] / den[ok]; pf = bs(p1, nb)[ok] / den[ok]
        el = np.hypot(bs(g * c2, nb)[ok], bs(g * s2, nb)[ok]) / den[ok]
        elm = bs(g * np.hypot(c2, s2), nb)[ok] / den[ok]
        d[f'lg{nb}_std'] = lg.std(); d[f'lg{nb}_max'] = lg.max() - lg.mean(); d[f'lg{nb}_min'] = lg.mean() - lg.min()
        d[f'pf{nb}_std'] = pf.std(); d[f'pf{nb}_max'] = pf.max()
        d[f'el{nb}_std'] = el.std(); d[f'el{nb}_mean'] = el.mean(); d[f'elm{nb}_std'] = elm.std()
        d[f'cor_pf_lg{nb}'] = np.corrcoef(pf, lg)[0, 1] if pf.std() > 1e-6 else 0
        ang = 0.5 * np.arctan2(bs(g * s2, nb)[ok], bs(g * c2, nb)[ok]); d[f'angdisp{nb}'] = 1 - np.hypot(np.cos(2 * ang).mean(), np.sin(2 * ang).mean())
    # phase vs matrix sizes and elongations
    d['la_ph_mx'] = (p1 * la).sum() / p1.sum() - (p0 * la).sum() / p0.sum()
    d['el_ph_mx'] = (p1 * np.hypot(c2, s2)).sum() / p1.sum() - (p0 * np.hypot(c2, s2)).sum() / p0.sum()
    d['hp_ph'] = -2 * np.log((p1 * np.exp(-0.5 * la)).sum() / p1.sum()); d['hp_mx'] = -2 * np.log((p0 * np.exp(-0.5 * la)).sum() / p0.sum())
    # phase clustering: smoothed phase field std at scales
    for s in (4, 8, 16):
        f = ndi.gaussian_filter(p1, s) / np.maximum(ndi.gaussian_filter(g, s), 1e-3); d[f'phf{s}_std'] = f.std()
    d['la_q90_q10'] = np.quantile(la[g > 0.5], 0.9) - np.quantile(la[g > 0.5], 0.1)
    d['la_skew'] = ((la[g > 0.5] - la[g > 0.5].mean()) ** 3).mean() / la[g > 0.5].std() ** 3
    d['bnd'] = p3.mean(); d['pore_n'] = ndi.label(p2 > 0.5)[1]
    return d
Fs = [feats(M['maps'][mi[i]].astype(np.float32)) for i in ids]
keys = list(Fs[0]); X = np.array([[f[k] for k in keys] for f in Fs])
np.savez(_repo_io(f'scan-{net}.npz'), ids=ids, keys=keys, X=X)
out = []
for j, k in enumerate(keys):
    v = np.nan_to_num(X[:, j]); out.append((k, spearmanr(v, r)[0], *[spearmanr(v[t == q], r[t == q])[0] for q in range(3)], spearmanr(v, y)[0]))
for o in sorted(out, key=lambda z: -abs(z[1])): print(f'{o[0]:12s} all {o[1]:+.3f} | {o[2]:+.3f} {o[3]:+.3f} {o[4]:+.3f} | y {o[5]:+.3f}')
