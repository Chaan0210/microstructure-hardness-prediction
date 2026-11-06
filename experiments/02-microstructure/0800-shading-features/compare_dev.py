"""Compare 039 base vs +1 column conditions (dev600 outputs), fixed W0.75, 4-variant mean, pooled over 3 partitions.
Segments: SCREENED = the 97-ish dev images used to pick the feature (noise>=11.9 gray, N_unet<300, noisy half) vs NOT-SCREENED (the rest);
NOISY-HALF = rows with ft-H300 masks (never in any auxiliary-model training originals). ID bootstrap 20,000 for the RMSE difference."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
E = str(_repo_path(_REPO_ROOT, 'experiments')) + '/'
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; parts = ['original', '20261031', '20261101']
conds = {'base': [E + f'push-600/dev-{v}-angh.npz' for v in V]}
for c in sys.argv[1:] or ['cor', 'raw', 'rnd']: conds[c] = [E + f'push-800/dev-{v}-angh{c}.npz' for v in V]
F = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
S = np.load(_repo_io(E + 'push-800/shade-train.npz'), allow_pickle=True); spos = {str(i): k for k, i in enumerate(S['ids'])}
ref = None
def se_of(files, W=0.75):
    global ref
    D = [np.load(_repo_io(f), allow_pickle=True) for f in files]
    if ref is None: ref = [str(i) for i in D[0]['original_ids']]
    out = []; per = []
    for n in parts:
        ids = [str(i) for i in D[0][f'{n}_ids']]; ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]
        y, p, f = D[0][f'{n}_y'][j], D[0][f'{n}_p021'][j], D[0][f'{n}_failed'][j]; g = np.mean([d[f'{n}_g'] for d in D], 0)[j]
        e = ((1 - W) * p + W * np.where(f, p, g) - y) ** 2; out.append(e); per.append(np.sqrt(e.mean()))
    return np.array(out), per
SE = {c: se_of(fs) for c, fs in conds.items()}
X = F['X'][[pos[i] for i in ref]].astype(float); nz = X[:, keys.index('noise')] * 255; N = 65536 / np.exp(X[:, keys.index('m_la_mean')])
src = S['mask_source'][[spos[i] for i in ref]]
noisy_half = src == 'H300'; screened = noisy_half & (nz >= 11.9) & (N < 300)
segs = {'ALL': np.ones(len(ref), bool), 'SCREENED': screened, 'NOT-SCREENED': ~screened, 'NOISY-HALF': noisy_half, 'NOISY-HALF minus screened': noisy_half & ~screened, 'CLEAN-HALF': ~noisy_half}
rng = np.random.RandomState(0); B = rng.randint(0, len(ref), (20000, len(ref)))
for c in conds:
    print(f'{c:5s} per-partition W0.75 ' + ' '.join(f'{v:.4f}' for v in SE[c][1]) + f'  pooled {np.sqrt(SE[c][0].mean()):.4f}')
for c in [k for k in conds if k != 'base']:
    print(f'--- {c} vs base (positive = improvement)')
    for sn, m in segs.items():
        a = SE['base'][0][:, m].mean(0); b = SE[c][0][:, m].mean(0); idx = np.where(m)[0]
        d = np.sqrt(a.mean()) - np.sqrt(b.mean())
        bs = []
        for row in B[:4000]:
            s = idx[np.searchsorted(np.sort(idx), idx)] if False else None
        # bootstrap within segment
        Bs = rng.randint(0, len(idx), (20000, len(idx)))
        bs = np.sqrt(a[Bs].mean(1)) - np.sqrt(b[Bs].mean(1))
        per = [np.sqrt(SE['base'][0][k, m].mean()) - np.sqrt(SE[c][0][k, m].mean()) for k in range(3)]
        print(f'  {sn:26s} n={m.sum():3d}  base {np.sqrt(a.mean()):7.4f} -> {np.sqrt(b.mean()):7.4f}  gain {d:+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  per-part ' + ' '.join(f'{v:+.3f}' for v in per))
