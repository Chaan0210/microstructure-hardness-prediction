"""053 structure with (a) TabPFN slots replaced (SLOT:template args) and/or (b) Ridge/logRidge replaced on noise>=12 rows by sr-<PART>.npz.
usage: eval_combo.py PART|none [SLOT:template ...]  (template uses {v}); slot replacement can be restricted to noise>=12 rows with env SLOTGATE=1"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sys.argv_ = sys.argv; exec(open(_repo_io('eval053.py')).read().split("base = {s: slot(L(f))")[0].split('"""', 2)[2])
nz = np.nan_to_num(np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['noise'], nan=0); gate = (nz >= 12) & ~B['fail'][0]
base = {s: slot(L(f)) for s, f in BASE.items()}
def build(slots, part):
    g = np.mean([slots[s] for s in ('38', '39', '44')], 0); r, l = B['ridge'], B['logridge']
    if part != 'none':
        S = np.load(_repo_io(f'sr-{part}.npz')); r = np.where(gate[None], S['ridge'], r); l = np.where(gate[None], S['logridge'], l)
    c = 0.16 * B['p021'] + 0.384 * g + 0.128 * r + 0.128 * l + 0.05 * B['set'] + 0.15 * B['gmm']; return np.where(B['fail'], B['p021'], c)
def rep(tag, a):
    e = a - y; print(f'{tag:44s} {np.sqrt((e ** 2).mean()):.4f} parts {np.round(np.sqrt((e ** 2).mean(1)), 4)} ' + ' '.join(f'{b} {np.sqrt((e[:, m] ** 2).mean()):.3f}' for b, m in [('clean', nz < 9.5), ('mid', (nz >= 9.5) & (nz < 12)), ('noisy', nz >= 12)]))
part = sys.argv[1]; new = dict(base)
for a in sys.argv[2:]:
    s, t = a.split(':', 1); rep_ = slot(L([t.format(v=v) for v in V]))
    sg = (nz >= float(os.environ.get('SLOTMIN', '12'))) & ~B['fail'][0]
    new[s] = np.where(sg[None], rep_, base[s]) if os.environ.get('SLOTGATE') == '1' else rep_
rep('053', build(base, 'none')); rep(f'partners {part}', build(base, part)); rep(f'slots replaced', build(new, 'none')); rep(f'slots + partners', build(new, part))
