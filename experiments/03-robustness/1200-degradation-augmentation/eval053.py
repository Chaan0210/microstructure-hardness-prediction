"""Evaluate replacements of TabPFN grain slots inside the exact 053 structure (stored dev OOF, Study900 ids/partitions):
053 = 0.16*p021 + 0.384*mean(slot038, slot039, slot044) + 0.128*ridge + 0.128*logRidge + 0.05*set(3 seeds) + 0.15*gmm  (failed -> p021).
usage: eval053.py SLOT:path_template [SLOT:path_template ...]   SLOT in 38|39|44, template with {v} for the 4 U-Net variants, e.g.
       eval053.py 44:dev-{v}-anghcorZS.npz      (several args = several slots replaced at once)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900'))
from blend_dev import load
E = '../'; V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; parts = ['original', '20261031', '20261101']
L = lambda fs: [np.load(_repo_io(f), allow_pickle=True) for f in fs]
BASE = {'38': [E + f'push-300/dev-{v}-v35.npz' for v in V], '39': [E + f'push-600/dev-{v}-angh.npz' for v in V], '44': [E + f'push-800/dev-{v}-anghcor.npz' for v in V]}
B = load(3); sid = B['ids']; y = B['y']
def slot(Ds):
    out = np.zeros((3, 400))
    for k, n in enumerate(parts):
        ids = [str(i) for i in Ds[0][f'{n}_ids']]; pos = {i: j for j, i in enumerate(ids)}; jj = [pos[i] for i in sid]
        assert np.allclose(Ds[0][f'{n}_y'][jj], y)
        out[k] = np.mean([d[f'{n}_g'] for d in Ds], 0)[jj]
    return out
def c053(slots):
    g = np.mean([slots[s] for s in ('38', '39', '44')], 0)
    c = 0.16 * B['p021'] + 0.384 * g + 0.128 * B['ridge'] + 0.128 * B['logridge'] + 0.05 * B['set'] + 0.15 * B['gmm']
    return np.where(B['fail'], B['p021'], c), g
def rmse_parts(a): return np.sqrt(((a - y) ** 2).mean(1)), float(np.sqrt(((a - y) ** 2).mean()))
base = {s: slot(L(f)) for s, f in BASE.items()}; cb, gb = c053(base); pb, rb = rmse_parts(cb)
print(f'053 base {rb:.4f} parts {np.round(pb, 4)}   grain block {rmse_parts(gb)[1]:.4f}')
if len(sys.argv) > 1:
    new = dict(base); lab = []
    for a in sys.argv[1:]:
        s, t = a.split(':', 1); new[s] = slot(L([t.format(v=v) for v in V])); lab.append(a)
        print(f'  slot {s} alone: base {rmse_parts(base[s])[1]:.4f} -> new {rmse_parts(new[s])[1]:.4f}')
    cn, gn = c053(new); pn, rn = rmse_parts(cn)
    print(f'{" + ".join(lab)}: 053 {rn:.4f} (gain {rb - rn:+.4f}; parts {np.round(pb - pn, 3)})  grain block {rmse_parts(gn)[1]:.4f}')
