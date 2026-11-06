"""Stripe-uniform dev (SU-dev): each measured-noise stripe (presets are discrete) gets equal total weight among dev noisy rows -- a train-only
robustness criterion (unknown test preset mix -> uniform), plus leave-one-stripe-out worst case. Diagnostic comparison with Public readings."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, nsgain as N
from blend_dev import load, align_dev
y, g, c61, nz = N.y, N.g, N.c61, N.nz; B = load(3); nzr = np.round(nz, 1)
st = sorted(set(nzr[g])); w = np.zeros(400)
for s in st: m = g & (nzr == s); w[m] = (g.sum() / len(st)) / m.sum()
def su(base, new, nm, pub=''):
    d = ((new - y) ** 2 - (base - y) ** 2)[:, g].mean(0); ws = w[g]; m = (ws * d).sum() / ws.sum()
    per = {s: d[nzr[g] == s].mean() for s in st}; loo = [(ws * d)[nzr[g] != s].sum() / ws[nzr[g] != s].sum() for s in st]
    se = np.sqrt(((ws * (d - m)) ** 2).sum()) / ws.sum()
    print(f'{nm:12s} plain {d.mean():+6.2f} | SU {m:+6.2f} (SE {se:.2f}) | LOSO worst {max(loo):+6.2f} | stripes improved {sum(v < 0 for v in per.values())}/{len(st)}   {pub}')
c53 = np.where(N.fail, B['p021'], 0.16 * B['p021'] + 0.384 * B['g'] + 0.128 * B['ridge'] + 0.128 * B['logridge'] + 0.05 * B['set'] + 0.15 * B['gmm'])
def part(t, base=c53, K=1.0):
    S = np.load(_repo_io(f'../push-1200/sr-{t}.npz')); return base + K * 0.128 * g * ((S['ridge'] - B['ridge']) + (S['logridge'] - B['logridge']))
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
def ta(tags):
    bs = {'38': '../push-300/dev-{v}-v35.npz', '39': '../push-600/dev-{v}-angh.npz', '44': '../push-800/dev-{v}-anghcor.npz'}
    return sum(0.128 * (np.mean([align_dev(f'../push-1200/dev-{v}-{t}.npz') for v in V], 0) - np.mean([align_dev(bs[s].format(v=v)) for v in V], 0)) for s, t in tags)
c57 = part('A-d23'); c62 = part('A-d1to7n12-wal'); c69 = np.where(g, c61 + ta([('38', 'v35TA4'), ('39', 'anghTA4'), ('44', 'anghcorTA4')]), c61)
c67 = np.where(g, c61 + ta([('38', 'v35TA'), ('39', 'anghTA'), ('44', 'anghcorTA')]), c61); c64 = np.where(g, c62 + (c67 - c61), c62)
aw = 0.256; C = N.C; prt = (C['ridge'] + C['logr']) / 2; oth = (c61 - aw * prt) / (1 - aw); k2 = np.where(g, (1 - 2 * aw) * oth + 2 * aw * prt, c61)
su(c53, c57, '053->057', 'Public -0.042'); su(c53, c61, '053->061', 'Public -0.055'); su(c57, c61, '057->061', 'Public -0.013')
su(c61, c64, '061->064', 'Public +0.034'); su(c61, c69, '061->069', 'Public +0.019'); su(c61, c67, '061->067', 'n/a'); su(c61, c62, '061->062', 'n/a'); su(c61, k2, '061->K2', 'n/a')
np.save(_repo_io('su_weights.npy'), w)
