"""Candidate 'nb' on dev: on non-gated rows (noise < 12, ~failed) the TabPFN block (12 members) and the three plain linear partners use features
without the het+intensity groups; gated rows unchanged. Reports TabPFN-only, partners-only, both; per band; bootstrap; per-slot."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import align_dev
import nsgain as N
from robust_eval import report
y, g, c61, C, nz, fail = N.y, N.g, N.c61, N.C, N.nz, N.fail; f = fail[0]; ng = ~g & ~f
S = ['038', '039', '044']; V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
M = {(s, v): align_dev(f'abl/dev-nb-{s}-{v}.npz') for s in S for v in V}; Tn = np.mean(list(M.values()), 0)
Z = np.load(_repo_io('nb-partners.npz'))
def build(tab=True, part=True):
    Cx = dict(C)
    if tab: Cx['tab'] = np.where(ng, Tn, C['tab'])
    if part:
        for nm, key in (('ridge', 'ridge'), ('logr', 'logridge'), ('gmm', 'gmm')): Cx[nm] = np.where(ng, C[nm] + (Z[key + '_nb_dev'] - Z[key + '_full_dev']), C[nm])
    return N.blend(Cx)
rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
print('TabPFN block non-gated RMSE: full %.3f nb %.3f | clean<11 %.3f -> %.3f | 11-12 %.3f -> %.3f' % (rb(C['tab'], ng), rb(Tn, ng), rb(C['tab'], (nz < 11) & ~f), rb(Tn, (nz < 11) & ~f), rb(C['tab'], ng & (nz >= 11)), rb(Tn, ng & (nz >= 11))))
rng = np.random.RandomState(0); BI = [rng.randint(0, 400, 400) for _ in range(4000)]
for nm, new in (('TabPFN only', build(True, False)), ('partners only', build(False, True)), ('both (candidate)', build(True, True))):
    d = (new - y) ** 2 - (c61 - y) ** 2; dm = d.mean(0)
    print(f'{nm:18s} non-gated dMSE {d[:, ng].mean():+.2f} ± {d[:, ng].mean(0).std() / np.sqrt(ng.sum()):.2f} | clean<11 {d[:, (nz < 11) & ~f].mean():+.2f} 11-12 {d[:, ng & (nz >= 11)].mean():+.2f} | P(better) {np.mean([dm[b].mean() < 0 for b in BI]):.3f}')
    report(f'   {nm}', new)
for s in S:
    Cx = dict(C); Cx['tab'] = np.where(ng, np.mean([M[(t, v)] if t == s else align_dev({'038': '../push-300/dev-{v}H2Z-v35.npz', '039': '../push-600/dev-{v}H2Z-angh.npz', '044': '../push-800/dev-{v}H2Z-anghcor.npz'}[t].format(v=v)) for t in S for v in V], 0), C['tab'])
    d = (N.blend(Cx) - y) ** 2 - (c61 - y) ** 2; print(f'   only slot {s} nb: non-gated dMSE {d[:, ng].mean():+.2f}')
np.save(_repo_io('dev-nb-candidate.npy'), build(True, True))
