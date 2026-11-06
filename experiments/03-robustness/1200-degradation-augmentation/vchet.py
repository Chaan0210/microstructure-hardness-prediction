"""Variance-component (random-effects ANOVA) estimator of zone-level grain-size heterogeneity from U-Net maps.
The generator scales grain size per 64-px zone (4x4 grid aligned to the image). The plain het4_u = std over zones of the block Hall-Petch
statistic b = -2 log(sum g a^-1/2 / sum g) is inflated by measurement noise and grain sampling inside each zone. Here each zone is split into
n x n sub-blocks; MS_B (between zones) and MS_W (within zones) give sigma2_B = (MS_B - MS_W)/n^2, an estimate of the between-zone variance with
the within-zone noise removed. Label-free, per image, no fitted parameters.
usage: vchet.py MAPFILE_TEMPLATE NETS OUT   e.g. vchet.py ../push-400/unet2maps-{net}-{split}.npz m1,m2,h1,m3 vchet  -> vchet-{train,test}.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
tmpl, nets, out = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
splits = sys.argv[4].split(',') if len(sys.argv) > 4 else ['train', 'test']
KEYS = []
def block_b(g, e, nb):
    s = 128 // nb; num = (g * e).reshape(nb, s, nb, s).sum((1, 3)); den = g.reshape(nb, s, nb, s).sum((1, 3)); ok = den > 0.2 * s * s
    b = np.full((nb, nb), np.nan); b[ok] = -2 * np.log(num[ok] / den[ok]); return b
def vc(bsub, n):
    # bsub: (4n, 4n) sub-block stats -> zones 4x4 with n x n sub-blocks
    B = bsub.reshape(4, n, 4, n).transpose(0, 2, 1, 3).reshape(16, n * n)
    ok = np.isfinite(B); cnt = ok.sum(1); keep = cnt >= max(2, (n * n) // 2)
    if keep.sum() < 4: return np.nan, np.nan, np.nan
    zm = np.array([np.nanmean(B[z]) for z in range(16)])[keep]; cz = cnt[keep]
    msw = np.sum([np.nansum((B[z] - np.nanmean(B[z])) ** 2) for z in np.where(keep)[0]]) / max(np.sum(cz - 1), 1)
    nbar = cz.mean(); gm = np.average(zm, weights=cz)
    msb = np.sum(cz * (zm - gm) ** 2) / (len(zm) - 1)
    s2b = (msb - msw) / nbar
    return s2b, msw, np.var(zm, ddof=1)
def feats(m):
    g = (m[0] + m[1]).astype(np.float64); e = np.exp(-0.5 * m[4].astype(np.float64)); o = {}
    b4 = block_b(g, e, 4); o['het4_re'] = np.nanstd(b4)
    for n in (2, 4):
        s2b, msw, vz = vc(block_b(g, e, 4 * n), n)
        o[f'vc{n}_s2b'] = s2b; o[f'vc{n}_sd'] = np.sign(s2b) * np.sqrt(abs(s2b)) if np.isfinite(s2b) else np.nan; o[f'vc{n}_msw'] = msw; o[f'vc{n}_vz'] = vz
    # same on log-area mean (geometric) statistic, pixel-weighted
    la = m[4].astype(np.float64)
    for n in (2, 4):
        nb = 4 * n; s = 128 // nb; num = (g * la).reshape(nb, s, nb, s).sum((1, 3)); den = g.reshape(nb, s, nb, s).sum((1, 3)); ok = den > 0.2 * s * s
        L = np.full((nb, nb), np.nan); L[ok] = num[ok] / den[ok]; s2b, msw, vz = vc(L, n)
        o[f'vcl{n}_sd'] = np.sign(s2b) * np.sqrt(abs(s2b)) if np.isfinite(s2b) else np.nan; o[f'vcl{n}_msw'] = msw
    return o
for split in splits:
    D = [np.load(_repo_io(tmpl.format(net=n, split=split))) for n in nets]; ids = D[0]['ids']; assert all((d['ids'] == ids).all() for d in D)
    rows = []
    for i in range(len(ids)):
        fs = [feats(d['maps'][i]) for d in D]; KEYS = list(fs[0]); rows.append([np.nanmean([f[k] for f in fs]) for k in KEYS])
        if i % 100 == 0: print(split, i, flush=True)
    np.savez(_repo_io(f'{out}-{split}.npz'), ids=ids, X=np.array(rows), keys=np.array(KEYS)); print(split, 'done', np.array(rows).shape, flush=True)
