"""Per-grain interior intensity on clean train images: is phase darkness an image-level level or per-grain continuum? relation to 039 residual."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from PIL import Image
P = str(_REPO_ROOT) + '/'
E = P + 'experiments/'
M = np.load(_repo_io(E + 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); mid = {str(i): k for k, i in enumerate(M['ids'])}
F = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X']; ids = [str(i) for i in F['ids']]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io(P + 'data/train.csv')))}
nz = X[:, keys.index('noise')] * 255
# residual: 039 grain OOF (original partition) for dev ids
D = [np.load(_repo_io(E + f'push-600/dev-{v}H2Z-angh.npz'), allow_pickle=True) for v in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4')]
rid = [str(i) for i in D[0]['original_ids']]; g = np.mean([d['original_g'] for d in D], 0); res = dict(zip(rid, D[0]['original_y'] - g))
order = np.argsort(nz)
rows = []
for k in order[:200]:
    i = ids[k]
    im = np.asarray(Image.open(_repo_io(P + f'data/train/{i}.png')), np.float32)
    mask = M['masks'][mid[i]].astype(int)
    er = ndi.minimum_filter(mask, 5) == ndi.maximum_filter(mask, 5)
    inner = np.where(er, mask, 0)
    labs = np.unique(inner); labs = labs[labs > 0]
    if len(labs) < 20: continue
    I = ndi.median(im, inner, labs); A = ndi.sum(np.ones_like(im), mask, labs)
    Icore = np.asarray(I); A = np.asarray(A)
    mat = np.average(Icore[Icore > np.percentile(Icore, 40)], weights=A[Icore > np.percentile(Icore, 40)])  # matrix level ~ bright grains
    d = mat - Icore   # darkness in gray levels
    rows.append(dict(id=i, y=lab[i], noise=nz[k], n=len(labs), mat=mat, d=d, A=A, img_p1=np.percentile(im, 0.5)))
# per image: sorted darkness profile
print('example darkness profiles (gray levels, sorted, every 10th pct):')
for r in rows[:12]:
    q = np.percentile(r['d'], [10, 30, 50, 60, 70, 80, 90, 95, 99])
    print(r['id'], 'y', round(r['y'], 1), 'n', r['n'], 'mat', round(r['mat'], 1), np.round(q, 1))
# histogram of darkness pooled (area weighted)
alld = np.concatenate([r['d'] for r in rows]); allA = np.concatenate([r['A'] for r in rows])
h, e = np.histogram(alld, bins=np.arange(-10, 80, 2.5), weights=allA)
print('pooled darkness hist (area):')
for a, b in zip(e[:-1], h / h.sum()): print(f'{a:6.1f} {"#" * int(b * 300)}')
# image-level stats
def stat(r):
    d, A = r['d'], r['A']; ph = d > 12
    f = A[ph].sum() / A.sum()
    pc = np.average(d[ph], weights=A[ph]) if ph.any() else np.nan
    pcs = np.sqrt(np.average((d[ph] - pc) ** 2, weights=A[ph])) if ph.sum() > 2 else np.nan
    msd = np.sqrt(np.average(d[~ph] ** 2, weights=A[~ph]))
    return f, pc, pcs, msd
S = np.array([stat(r) for r in rows]); yy = np.array([r['y'] for r in rows]); rr = np.array([res.get(r['id'], np.nan) for r in rows])
ok = ~np.isnan(rr)
for j, nm in enumerate(['phase_frac(d>12)', 'phase_contrast', 'phase_contrast_sd', 'matrix_sd']):
    v = S[:, j]; m = ~np.isnan(v)
    print(f'{nm:20s} rho(y) {spearmanr(v[m], yy[m])[0]:+.3f}   rho(039 resid) {spearmanr(v[m & ok], rr[m & ok])[0]:+.3f}  (n={int((m & ok).sum())})  mean {np.nanmean(v):.2f} sd {np.nanstd(v):.2f}')
# partial: contrast vs y controlling phase frac (linear)
m = ~np.isnan(S).any(1)
A_ = np.c_[np.ones(m.sum()), S[m, 0]]
ry = yy[m] - A_ @ np.linalg.lstsq(A_, yy[m], rcond=None)[0]; rc = S[m, 1] - A_ @ np.linalg.lstsq(A_, S[m, 1], rcond=None)[0]
print('partial corr(contrast, y | frac)', round(np.corrcoef(ry, rc)[0, 1], 3))
np.save(_repo_io('/private/tmp/claude-501/-Users-dongchan-Desktop-PythonWorkspace-microstructure-2026/f62e347b-cb1a-4c17-8d54-2bf505be6706/scratchpad/p1_rows.npy'), np.array(rows, dtype=object), allow_pickle=True)
