"""Cross-fitted slope stretch of the 045 prediction on the low-noise half of the high-noise group (idea from an external repo's public notes:
their OOF slope there was 1.49). Rule fixed before looking at our gains: group = noise >= 12 (features-v3 noise x 255), half split at the
TRAIN median of noise inside the group (label-free), p' = m + k (p - m) with m = mean prediction and b = OLS slope of y on p, both from the
out-of-fold rows of the same partition only; k = 1 + lam (b - 1), lam in {0.5, 1}. Pooled dev over the 3 partitions x 5 folds, ID bootstrap.
usage: stretch_dev.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
E = '../'; V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; parts = ['original', '20261031', '20261101']
L = lambda fs: [np.load(_repo_io(f), allow_pickle=True) for f in fs]
D38 = L([E + f'push-300/dev-{v}-v35.npz' for v in V]); D39 = L([E + f'push-600/dev-{v}-angh.npz' for v in V]); D44 = L([f'dev-{v}-anghcor.npz' for v in V]); ref = D39[0]
G = lambda Ds, n: np.mean([d[f'{n}_g'] for d in Ds], 0)
F = np.load(_repo_io(E + 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); fk = list(F['keys']); nzall = F['X'][:, fk.index('noise')] * 255
nzmap = {str(i): x for i, x in zip(F['ids'], nzall)}
med = float(np.median(nzall[nzall >= 12])); print(f'train group noise >= 12: n={int((nzall >= 12).sum())}, median {med:.2f}')
oid = [str(i) for i in ref['original_ids']]
def run(lo, hi, lam):
    base, new = [], []
    for n in parts:
        ids = [str(i) for i in ref[f'{n}_ids']]; y = ref[f'{n}_y']; q = ref[f'{n}_p021']; f = ref[f'{n}_failed']; fold = ref[f'{n}_fold']
        g = np.where(f, q, np.mean([G(D38, n), G(D39, n), G(D44, n)], 0)); p = 0.2 * q + 0.8 * g; nz = np.array([nzmap[i] for i in ids])
        band = (nz >= lo) & (nz < hi); pn = p.copy()
        for k in np.unique(fold):
            tr = band & (fold != k); te = band & (fold == k)
            b = np.polyfit(p[tr], y[tr], 1)[0]; m = p[tr].mean(); kk = 1 + lam * (b - 1); pn[te] = m + kk * (p[te] - m)
        jj = [ids.index(i) for i in oid]; base.append(((p - y) ** 2)[jj]); new.append(((pn - y) ** 2)[jj])
    return np.array(base), np.array(new)
rng = np.random.RandomState(0)
for nm, lo, hi in (('low half', 12, med), ('high half', med, 99), ('whole group', 12, 99)):
    for lam in (0.5, 1.0):
        b, s = run(lo, hi, lam); rb, rs = np.sqrt(b.mean()), np.sqrt(s.mean())
        d = b.mean(0) - s.mean(0); boot = [np.sqrt(b.mean(0)[ix].mean()) - np.sqrt(s.mean(0)[ix].mean()) for ix in (rng.randint(0, 400, 400) for _ in range(2000))]
        print(f'{nm:11s} lam {lam:.1f}: 045 {rb:.4f} -> {rs:.4f}  gain {rb - rs:+.4f}  per-part ' + ' '.join(f'{np.sqrt(b[k].mean()) - np.sqrt(s[k].mean()):+.3f}' for k in range(3))
              + f'  95% CI [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]  rows changed {int((d != 0).sum())}')
