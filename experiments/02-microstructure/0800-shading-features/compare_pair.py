"""Pairwise dev comparison (fixed W0.75, 4-variant mean, pooled over 3 partitions) of push-800 dev tags, with segment breakdown and ID bootstrap.
usage: compare_pair.py BASE_TAG COND_TAG [COND_TAG ...]   (tag 'b039' = push-600 039 files)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
E = str(_repo_path(_REPO_ROOT, 'experiments')) + '/'
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; parts = ['original', '20261031', '20261101']
def files(t): return [E + (f'push-600/dev-{v}-angh.npz' if t == 'b039' else f'push-800/dev-{v}-angh{t}.npz') for v in V]
def se(t):
    D = [np.load(_repo_io(f), allow_pickle=True) for f in files(t)]; ref = [str(i) for i in D[0]['original_ids']]; out = []
    for n in parts:
        ids = [str(i) for i in D[0][f'{n}_ids']]; ix = {i: k for k, i in enumerate(ids)}; j = [ix[i] for i in ref]
        y, p, f = D[0][f'{n}_y'][j], D[0][f'{n}_p021'][j], D[0][f'{n}_failed'][j]; g = np.mean([d[f'{n}_g'] for d in D], 0)[j]
        out.append((0.25 * p + 0.75 * np.where(f, p, g) - y) ** 2)
    return np.array(out), ref
F = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
S = np.load(_repo_io(E + 'push-800/shade-train.npz'), allow_pickle=True); spos = {str(i): k for k, i in enumerate(S['ids'])}
b, ref = se(sys.argv[1])
X = F['X'][[pos[i] for i in ref]].astype(float); noisy = S['mask_source'][[spos[i] for i in ref]] == 'H300'
segs = {'ALL': np.ones(len(ref), bool), 'CLEAN-HALF': ~noisy, 'NOISY-HALF': noisy}
rng = np.random.RandomState(0)
print(f'{sys.argv[1]:8s} pooled {np.sqrt(b.mean()):.4f} per-part ' + ' '.join(f'{np.sqrt(b[k].mean()):.3f}' for k in range(3)))
for t in sys.argv[2:]:
    c, _ = se(t)
    print(f'{t:8s} pooled {np.sqrt(c.mean()):.4f} per-part ' + ' '.join(f'{np.sqrt(c[k].mean()):.3f}' for k in range(3)))
    for sn, m in segs.items():
        a_, c_ = b[:, m].mean(0), c[:, m].mean(0); idx = rng.randint(0, m.sum(), (20000, m.sum()))
        bs = np.sqrt(a_[idx].mean(1)) - np.sqrt(c_[idx].mean(1))
        print(f'    {sn:10s} n={m.sum():3d} gain vs {sys.argv[1]} {np.sqrt(a_.mean()) - np.sqrt(c_.mean()):+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  per-part ' +
              ' '.join(f'{np.sqrt(b[k, m].mean()) - np.sqrt(c[k, m].mean()):+.3f}' for k in range(3)))
