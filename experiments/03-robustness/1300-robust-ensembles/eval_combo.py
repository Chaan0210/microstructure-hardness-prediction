"""Combined candidate on dev: 073 (nb on non-gated) + 074 (gated DTA logRidge alpha via source-grouped CV) + H9 (2 cumulant columns in all 12 TabPFN members,
nb+h9 on non-gated rows, full+h9 on gated rows). Reports each piece and the combination; image bootstrap; per band."""
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
full = {'038': '../push-300/dev-{v}H2Z-v35.npz', '039': '../push-600/dev-{v}H2Z-angh.npz', '044': '../push-800/dev-{v}H2Z-anghcor.npz'}
h9 = lambda fs, s, v: f'abl/dev-h9phys-{fs}-044-{v}.npz' if s == '044' else f'abl/dev-h9-{fs}-{s}-{v}.npz'
b0 = np.mean([np.where(ng, align_dev(f'abl/dev-nb-{s}-{v}.npz'), align_dev(full[s].format(v=v))) for s in S for v in V], 0)
b9 = np.mean([np.where(ng, align_dev(h9('nb', s, v)), align_dev(h9('full', s, v))) for s in S for v in V], 0)
c73 = np.load(_repo_io('dev-nb-candidate.npy')); Z = np.load(_repo_io('srcw.npz')); d74 = np.where(g, 0.128 * (Z['B_ridge'] - C['ridge']) + 0.128 * (Z['B_logridge'] - C['logr']), 0)
d9 = np.where(fail, 0, 0.384 * (b9 - b0))
rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
print('TabPFN block: non-gated %.3f -> %.3f | gated %.3f -> %.3f' % (rb(b0, ng), rb(b9, ng), rb(b0, g), rb(b9, g)))
rng = np.random.RandomState(0); BI = [rng.randint(0, 400, 400) for _ in range(4000)]
for nm, new, base in (('H9 on 073', c73 + d9, c73), ('074 on 073', c73 + d74, c73), ('H9+074 on 073 (combo)', c73 + d9 + d74, c73), ('combo vs 061', c73 + d9 + d74, N.c61)):
    d = ((new - y) ** 2 - (base - y) ** 2).mean(0); report(f'{nm}', new, base=base)
    print(f'      bands: clean<11 {((new - y) ** 2 - (base - y) ** 2)[:, (nz < 11) & ~f].mean():+.2f} 11-12 {((new - y) ** 2 - (base - y) ** 2)[:, ng & (nz >= 11)].mean():+.2f} noisy {((new - y) ** 2 - (base - y) ** 2)[:, g].mean():+.2f} | bootstrap P(better) {np.mean([d[b].mean() < 0 for b in BI]):.3f}')
np.save(_repo_io('dev-combo.npy'), c73 + d9 + d74); print('combo dev RMSE %.4f (073 %.4f, 061 %.4f)' % (N.rm(c73 + d9 + d74), N.rm(c73), N.rm(N.c61)))
