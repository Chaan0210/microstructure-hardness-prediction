"""Exact 045-structure dev (0.2*best021 + 0.8*mean grain(038, 039, X)), swapping only the 044 component X. usage: cmp045.py TAG [TAG ...] (push-800 dev tags; 'cor' = 044)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
E = '../'; V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; parts = ['original', '20261031', '20261101']
L = lambda fs: [np.load(_repo_io(f), allow_pickle=True) for f in fs]
D38 = L([E + f'push-300/dev-{v}-v35.npz' for v in V]); D39 = L([E + f'push-600/dev-{v}-angh.npz' for v in V]); ref = D39[0]; oid = [str(i) for i in ref['original_ids']]
G = lambda Ds, n: np.mean([d[f'{n}_g'] for d in Ds], 0)
def se(tag):
    DX = L([f'dev-{v}-angh{tag}.npz' for v in V]); out = []
    for n in parts:
        assert all((d[f'{n}_ids'] == ref[f'{n}_ids']).all() and np.allclose(d[f'{n}_p021'], ref[f'{n}_p021']) for d in DX)
        ids = [str(i) for i in ref[f'{n}_ids']]; ix = {i: j for j, i in enumerate(ids)}; jj = [ix[i] for i in oid]
        y, q, f = ref[f'{n}_y'], ref[f'{n}_p021'], ref[f'{n}_failed']; g = np.mean([G(D38, n), G(D39, n), G(DX, n)], 0)
        out.append(((0.2 * q + 0.8 * np.where(f, q, g) - y) ** 2)[jj])
    return np.array(out)
b = se('cor'); rb = np.sqrt(b.mean()); print(f'045 (X=044) {rb:.4f}')
for t in sys.argv[1:]:
    s = se(t); print(f'  X={t:8s} {np.sqrt(s.mean()):.4f}  gain {rb - np.sqrt(s.mean()):+.4f}  per-part ' + ' '.join(f'{np.sqrt(b[k].mean()) - np.sqrt(s[k].mean()):+.3f}' for k in range(3)))
