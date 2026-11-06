"""Exact 045-structure dev with ALL three grain slots swapped: 0.2*best021 + 0.8*mean(g038', g039', g044') vs 045 (stored OOF; same ids, folds, labels,
021 OOF and failure flags asserted). Per partition, per noise band (clean < 9.5 <= mid < 12 <= noisy, push-300 noise x 255), image-ID bootstrap.
usage: cmp045all.py TAG [TAG ...]   (push-900 dev-<v>{H2Z,H2Z-angh,H2Z-anghcor}-<TAG>.npz)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']; parts = ['original', '20261031', '20261101']
L = lambda fs: [np.load(_repo_io(f), allow_pickle=True) for f in fs]
B = [L([_repo_path(E, f'push-300/dev-{v}H2Z-v35.npz') for v in V]), L([_repo_path(E, f'push-600/dev-{v}H2Z-angh.npz') for v in V]), L([_repo_path(E, f'push-800/dev-{v}H2Z-anghcor.npz') for v in V])]
ref = B[1][0]; oid = [str(i) for i in ref['original_ids']]
F = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); nzmap = dict(zip([str(i) for i in F['ids']], np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255))
nz = np.array([nzmap[i] for i in oid]); bands = {'clean': nz < 9.5, 'mid': (nz >= 9.5) & (nz < 12), 'noisy': nz >= 12}
def sq(sets):
    out = []
    for n in parts:
        for Ds in sets:
            for d in Ds: assert (d[f'{n}_ids'] == ref[f'{n}_ids']).all() and np.allclose(d[f'{n}_p021'], ref[f'{n}_p021']) and np.allclose(d[f'{n}_y'], ref[f'{n}_y'])
        ids = [str(i) for i in ref[f'{n}_ids']]; jj = [ids.index(i) for i in oid]; y, q, f = ref[f'{n}_y'], ref[f'{n}_p021'], ref[f'{n}_failed']
        g = np.mean([np.mean([d[f'{n}_g'] for d in Ds], 0) for Ds in sets], 0)
        out.append(((0.2 * q + 0.8 * np.where(f, q, g) - y) ** 2)[jj])
    return np.array(out)
b = sq(B); rb = np.sqrt(b.mean()); print(f'045 baseline {rb:.4f}  bands ' + ' '.join(f'{k} {np.sqrt(b[:, m].mean()):.3f}' for k, m in bands.items()))
rng = np.random.RandomState(0); boots = [rng.randint(0, 400, 400) for _ in range(2000)]
for t in sys.argv[1:]:
    N = [L([_repo_path(H, f'dev-{v}{s}-{t}.npz') for v in V]) for s in ('H2Z', 'H2Z-angh', 'H2Z-anghcor')]
    s = sq(N); rs = np.sqrt(s.mean()); bm, sm = b.mean(0), s.mean(0)
    ci = np.percentile([np.sqrt(bm[ix].mean()) - np.sqrt(sm[ix].mean()) for ix in boots], [2.5, 97.5])
    print(f'  {t:12s} {rs:.4f} gain {rb - rs:+.4f} CI [{ci[0]:+.3f},{ci[1]:+.3f}] per-part ' + ' '.join(f'{np.sqrt(b[k].mean()) - np.sqrt(s[k].mean()):+.3f}' for k in range(3))
          + ' | bands ' + ' '.join(f'{k} {np.sqrt(b[:, m].mean()) - np.sqrt(s[:, m].mean()):+.3f}' for k, m in bands.items()))
    for j, nm in enumerate(('038', '039', '044')):
        M = [B[i] if i != j else N[j] for i in range(3)]; s1 = sq(M); print(f'      only {nm} slot swapped: gain {rb - np.sqrt(s1.mean()):+.4f}')
