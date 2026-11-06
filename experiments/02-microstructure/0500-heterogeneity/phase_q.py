"""Phase-estimator quality on REAL images: partial correlation of a phase estimate with hardness, controlling for non-phase structure
(U-Net mean size / size heterogeneity / elongation from the m-avg set + stcoh + pore), by noise half. Paired bootstrap vs a reference.
usage: phase_q.py MAPS_TAG[,MAPS_TAG..] [REF_TAG]   (maps from push-400/unet2maps-<TAG>-train.npz; several tags = averaged maps)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, '.')
import resid035 as R
lab = R.labels(); nz = R.noise()
F = np.load(_repo_io('../push-400/feat-u1dmavg-train.npz'), allow_pickle=True); k = [str(x) for x in F['keys']]; ids = [str(i) for i in F['ids']]
y = np.array([lab[i] for i in ids]); n = np.array([nz[i] for i in ids]) * 255; h = n >= np.median(n)
ctrl = ['m_la_mean', 'm_la_f32', 'm_la_tile8', 'm_el_glob', 'm_el_pix', 'stcoh_s3_w8_locmed', 'stcoh_s1.5_w32_glob', 'pore_frac', 'm_la_std']
Xc = F['X'][:, [k.index(c) for c in ctrl]].astype(float); Xc = np.where(np.isnan(Xc), np.nanmedian(Xc, 0), Xc)
C = np.c_[np.ones(len(y)), Xc]
def readouts(tag):
    tags = tag.split(','); maps = [dict(np.load(_repo_io(f'../push-400/unet4maps-{t[2:]}-train.npz' if t.startswith('u4') else f'../push-400/unet2maps-{t}-train.npz'))) for t in tags]
    for M in maps: assert [str(i) for i in M['ids']] == ids
    C4 = [M['maps'][:, :4] for M in maps]
    out = {'soft': [], 'gt05': [], 'argmax': []}
    for j in range(len(ids)):
        m = np.mean([c[j].astype(np.float32) for c in C4], 0); p0, p1, p2 = m[0], m[1], m[2]
        out['soft'].append(p1.sum() / (p0 + p1).sum()); out['gt05'].append((p1 > 0.5).mean() / (1 - p2).mean())
        am = m[:4].argmax(0); out['argmax'].append((am == 1).sum() / max(((am == 0) | (am == 1)).sum(), 1))
    return {a: np.array(b) for a, b in out.items()}
def pcorr(v, sel):
    A = C[sel]; ry = y[sel] - A @ np.linalg.lstsq(A, y[sel], rcond=None)[0]; rv = v[sel] - A @ np.linalg.lstsq(A, v[sel], rcond=None)[0]
    return np.corrcoef(rv, ry)[0, 1]
tags = sys.argv[1:]
res = {t: readouts(t) for t in tags}
rng = np.random.RandomState(0); boots = [rng.randint(0, h.sum(), h.sum()) for _ in range(300)]
hi = np.where(h)[0]; ci = np.where(~h)[0]
for t in tags:
    for a, v in res[t].items():
        print(f'{t:14s} {a:7s} clean-half pcorr {pcorr(v, ~h):+.3f} | noisy-half pcorr {pcorr(v, h):+.3f}  raw corr {np.corrcoef(v[h], y[h])[0, 1]:+.3f}')
if len(tags) > 1:
    ref = tags[0]
    for t in tags[1:]:
        for a in ('argmax', 'gt05', 'soft'):
            d = [pcorr(res[t][a], hi[b]) - pcorr(res[ref][a], hi[b]) for b in boots]
            print(f'noisy-half {a:7s} {t} - {ref}: {np.mean(d):+.3f}  (90% {np.percentile(d, 5):+.3f}, {np.percentile(d, 95):+.3f})')
