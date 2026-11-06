"""Evaluate a replacement for the 044 grain slot (dev-<v>-anghcor<TAG>.npz in push-900) in the 045 and 051 structures (stored OOF, Study900 ids).
usage: eval_slot44.py TAG [TAG ...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
E = '../'; V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; parts = ['original', '20261031', '20261101']
L = lambda fs: [np.load(_repo_io(f), allow_pickle=True) for f in fs]
D38 = L([E + f'push-300/dev-{v}-v35.npz' for v in V]); D39 = L([E + f'push-600/dev-{v}-angh.npz' for v in V]); D44 = L([E + f'push-800/dev-{v}-anghcor.npz' for v in V]); ref = D39[0]
G = lambda Ds, n: np.mean([d[f'{n}_g'] for d in Ds], 0)
S = np.load(_repo_io('../probabilistic-grains-900/ridge/predictions.npz'), allow_pickle=True); sid = [str(i) for i in S['ids']]; LG = np.load(_repo_io('diverse-ridge_a2_logy_var4.npz'))['oof']
def ev(Dx):
    out = {}
    for k, n in enumerate(parts):
        ids = [str(i) for i in ref[f'{n}_ids']]; jj = [ids.index(i) for i in sid]; y, q, f = ref[f'{n}_y'][jj], ref[f'{n}_p021'][jj], ref[f'{n}_failed'][jj]
        gx = G(Dx, n)[jj]; g = np.mean([G(D38, n)[jj], G(D39, n)[jj], gx], 0)
        for nm, pr in (('slot_alone', gx), ('045', 0.2 * q + 0.8 * g), ('051', 0.2 * q + 0.48 * g + 0.16 * S['ridge'][k] + 0.16 * LG[k])):
            out.setdefault(nm, []).append((np.where(f, q, pr) - y) ** 2)
    return {k: np.array(v) for k, v in out.items()}
b = ev(D44); print('base   ' + '  '.join(f'{k} {np.sqrt(v.mean()):.4f}' for k, v in b.items()))
for t in sys.argv[1:]:
    s = ev(L([f'dev-{v}-anghcor{t}.npz' for v in V]))
    print(f'{t:6s} ' + '  '.join(f'{k} {np.sqrt(v.mean()):.4f} ({np.sqrt(b[k].mean()) - np.sqrt(v.mean()):+.4f}; parts ' + '/'.join(f'{np.sqrt(b[k][i].mean()) - np.sqrt(v[i].mean()):+.3f}' for i in range(3)) + ')' for k, v in s.items()))
