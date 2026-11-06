"""H9 screen on the 044 slot: members with +2 cumulant columns (phys) or +2 random columns (rand), nb features for non-gated rows, full for gated rows.
Blend effect when only the 044 slot's 4 members change (other 8 members unchanged), on top of 073 (+ optional 074 partner change)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import align_dev
import nsgain as N
from robust_eval import report
y, g, C, fail, nz = N.y, N.g, N.C, N.fail, N.nz; f = fail[0]; ng = ~g & ~f
S = ['038', '039', '044']; V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
full = {(s, v): align_dev({'038': '../push-300/dev-{v}H2Z-v35.npz', '039': '../push-600/dev-{v}H2Z-angh.npz', '044': '../push-800/dev-{v}H2Z-anghcor.npz'}[s].format(v=v)) for s in S for v in V}
nb = {(s, v): align_dev(f'abl/dev-nb-{s}-{v}.npz') for s in S for v in V}
c73 = np.load(_repo_io('dev-nb-candidate.npy')); Z = np.load(_repo_io('srcw.npz')); c74 = np.where(g, c73 + 0.128 * (Z['B_ridge'] - C['ridge']) + 0.128 * (Z['B_logridge'] - C['logr']), c73)
def block(sub):
    mem = []
    for s in S:
        for v in V:
            base = np.where(ng, nb[(s, v)], full[(s, v)])
            if s == '044' and sub is not None: base = np.where(ng, align_dev(f'abl/dev-h9{sub}-nb-044-{v}.npz'), align_dev(f'abl/dev-h9{sub}-full-044-{v}.npz'))
            mem.append(base)
    return np.mean(mem, 0)
b0 = block(None); rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
for sub in ('phys', 'rand'):
    b = block(sub); print(f'{sub}: block RMSE non-gated {rb(b0, ng):.3f}->{rb(b, ng):.3f}, gated {rb(b0, g):.3f}->{rb(b, g):.3f}')
    for nm, base in (('073', c73), ('074', c74)):
        new = np.where(fail, base, base + 0.384 * (b - b0)); report(f'   044-slot {sub} on {nm}', new, base=base)
report('074 vs 073', c74, base=c73)
