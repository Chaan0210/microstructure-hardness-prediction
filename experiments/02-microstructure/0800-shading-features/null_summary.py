"""Null distribution of the dev gain from adding one unscreened N(0,1) column to 039 (4 variants, W0.75, pooled), vs sh_cor15 / sh_raw15."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import glob, numpy as np
E = str(_repo_path(_REPO_ROOT, 'experiments')) + '/'
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; parts = ['original', '20261031', '20261101']
def pooled(files):
    D = [np.load(_repo_io(f), allow_pickle=True) for f in files]; ref = [str(i) for i in D[0]['original_ids']]; se = []
    for n in parts:
        ids = [str(i) for i in D[0][f'{n}_ids']]; ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]
        y, p, f = D[0][f'{n}_y'][j], D[0][f'{n}_p021'][j], D[0][f'{n}_failed'][j]; g = np.mean([d[f'{n}_g'] for d in D], 0)[j]
        se.append((0.25 * p + 0.75 * np.where(f, p, g) - y) ** 2)
    return float(np.sqrt(np.mean(se)))
base = pooled([E + f'push-600/dev-{v}-angh.npz' for v in V])
res = {}
for tag in ['cor', 'raw', 'rnd'] + [f'rnd{k}' for k in range(1, 7)]:
    fs = [E + f'push-800/dev-{v}-angh{tag}.npz' for v in V]
    if all(__import__('os').path.exists(f) for f in fs): res[tag] = base - pooled(fs)
nulls = np.array([v for k, v in res.items() if k.startswith('rnd')])
print(f'base {base:.4f}')
for k, v in res.items(): print(f'{k:5s} gain {v:+.4f}')
if len(nulls) > 1:
    m, s = nulls.mean(), nulls.std(ddof=1)
    print(f'null (n={len(nulls)}): mean {m:+.4f} sd {s:.4f} max {nulls.max():+.4f}')
    for k in ('cor', 'raw'):
        if k in res: print(f'{k}: z vs null {(res[k] - m) / s:+.2f}, exceeds {np.mean(nulls < res[k]) * 100:.0f}% of nulls, gain over null mean {res[k] - m:+.4f}')
# segment-level: gain (RMSE base - cond) within SCREENED / NOT-SCREENED / CLEAN-HALF, cond vs null controls
F = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
S = np.load(_repo_io(E + 'push-800/shade-train.npz'), allow_pickle=True); spos = {str(i): k for k, i in enumerate(S['ids'])}
def se_rows(files):
    D = [np.load(_repo_io(f), allow_pickle=True) for f in files]; ref = [str(i) for i in D[0]['original_ids']]; se = []
    for n in parts:
        ids = [str(i) for i in D[0][f'{n}_ids']]; ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]
        y, p, f = D[0][f'{n}_y'][j], D[0][f'{n}_p021'][j], D[0][f'{n}_failed'][j]; g = np.mean([d[f'{n}_g'] for d in D], 0)[j]
        se.append((0.25 * p + 0.75 * np.where(f, p, g) - y) ** 2)
    return np.mean(se, 0), ref
b, ref = se_rows([E + f'push-600/dev-{v}-angh.npz' for v in V])
X = F['X'][[pos[i] for i in ref]].astype(float); nz = X[:, keys.index('noise')] * 255; N = 65536 / np.exp(X[:, keys.index('m_la_mean')])
noisy = S['mask_source'][[spos[i] for i in ref]] == 'H300'; scr = noisy & (nz >= 11.9) & (N < 300)
segs = {'SCREENED': scr, 'NOT-SCREENED': ~scr, 'CLEAN-HALF': ~noisy, 'NOISY-not-scr': noisy & ~scr}
G = {}
for tag in res:
    s, _ = se_rows([E + f'push-800/dev-{v}-angh{tag}.npz' for v in V]); G[tag] = {sn: np.sqrt(b[m].mean()) - np.sqrt(s[m].mean()) for sn, m in segs.items()}
for sn in segs:
    nl = np.array([G[t][sn] for t in G if t.startswith('rnd')])
    print(f'{sn:13s} n={segs[sn].sum():3d} null mean {nl.mean():+.4f} sd {nl.std(ddof=1):.4f} max {nl.max():+.4f} | cor {G["cor"][sn]:+.4f} (z {(G["cor"][sn] - nl.mean()) / nl.std(ddof=1):+.2f}) | raw {G["raw"][sn]:+.4f} (z {(G["raw"][sn] - nl.mean()) / nl.std(ddof=1):+.2f})')
