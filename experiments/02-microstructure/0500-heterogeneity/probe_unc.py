"""Probe: are U-Net 'uncertain' phase pixels (0.2<p_phase<0.7) real second phase in noisy images? Spearman with hardness / 035 residual by noise tercile."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr
sys.path.insert(0, '.')
import resid035 as R
ids, y, B, G, Pp = R.load(); b = B.mean(0); r = y - b
nz = R.noise(); n = np.array([nz[i] for i in ids]); t = np.digitize(n, np.quantile(n, [1/3, 2/3]))
M = np.load(_repo_io('../push-400/unet2maps-m1-train.npz')); mi = {str(i): k for k, i in enumerate(M['ids'])}
F = {}
for i in ids:
    m = M['maps'][mi[i]].astype(float); p0, p1, p2 = m[0], m[1], m[2]; g = p0 + p1
    im = np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(float)[::2, ::2]
    sm = ndi.gaussian_filter(np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(float), 2.5)[::2, ::2]
    d = {}
    d['soft'] = p1.sum() / g.sum()
    d['conf'] = (p1 > 0.7).mean() / (g > 0.5).mean()
    d['unc'] = ((p1 > 0.2) & (p1 < 0.7)).mean()
    d['conf+unc'] = d['conf'] + d['unc']
    d['hard'] = (p1 > p0).sum() / (g > 0.5).sum()
    # intensity calibration: matrix level = median sm where p0>0.8; phase level = median sm where p1>0.8 (fallback: matrix - 2*mad)
    mx = sm[p0 > 0.8]; ml = np.median(mx); ms = 1.4826 * np.median(np.abs(mx - ml))
    gl = (g > 0.5)
    d['int_thr2'] = (sm[gl] < ml - 2 * ms).mean()
    d['int_thr3'] = (sm[gl] < ml - 3 * ms).mean()
    if (p1 > 0.8).sum() > 30:
        pl = np.median(sm[p1 > 0.8]); thr = 0.5 * (ml + pl); d['int_mid'] = (sm[gl] < thr).mean(); d['gap_vs_mxsd'] = (ml - pl) / ms
    else:
        d['int_mid'] = np.nan; d['gap_vs_mxsd'] = np.nan
    F[i] = d
keys = list(next(iter(F.values())).keys())
print(f"{'feat':12s} " + ' '.join(f'{s:>14s}' for s in ['y clean', 'y mid', 'y noisy', 'res clean', 'res mid', 'res noisy']))
for k in keys:
    v = np.array([F[i][k] for i in ids])
    ry = [spearmanr(v[t == q], y[t == q], nan_policy='omit')[0] for q in range(3)]
    rr = [spearmanr(v[t == q], r[t == q], nan_policy='omit')[0] for q in range(3)]
    print(f'{k:12s} ' + ' '.join(f'{x:+14.3f}' for x in ry + rr))
np.savez(_repo_io('probe_unc.npz'), ids=ids, keys=keys, X=np.array([[F[i][k] for k in keys] for i in ids]))
