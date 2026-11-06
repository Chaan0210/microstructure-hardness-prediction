"""h9s2 (sigma 2/4 p34 + contrast) in all 12 TabPFN members on top of 078 dev; 044 slot from the screen files (identical features)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import align_dev
import nsgain as N
from robust_eval import report
y, g, fail, nz = N.y, N.g, N.fail, N.nz; f = fail[0]; ng = ~g & ~f; S = ['038', '039', '044']; V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
h9 = lambda fs, s, v: f'abl/dev-h9phys-{fs}-044-{v}.npz' if s == '044' else f'abl/dev-h9-{fs}-{s}-{v}.npz'
h9s = lambda fs, s, v: f'abl/dev-h9s-{fs}-044-{v}.npz' if s == '044' else f'abl/dev-h9s2-{fs}-{s}-{v}.npz'
b0 = np.mean([np.where(ng, align_dev(h9('nb', s, v)), align_dev(h9('full', s, v))) for s in S for v in V], 0)
b1 = np.mean([np.where(ng, align_dev(h9s('nb', s, v)), align_dev(h9s('full', s, v))) for s in S for v in V], 0)
c78 = np.load(_repo_io('dev-078.npy')); new = np.where(fail, c78, c78 + 0.384 * (b1 - b0)); np.save(_repo_io('dev-079.npy'), new)
rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
print('block non-gated %.3f -> %.3f | gated %.3f -> %.3f' % (rb(b0, ng), rb(b1, ng), rb(b0, g), rb(b1, g)))
report('079 vs 078', new, base=c78); report('079 vs 076', new, base=np.load(_repo_io('dev-combo.npy'))); report('079 vs 061', new, base=N.c61)
d = (new - y) ** 2 - (c78 - y) ** 2; print('bands vs 078: clean<11 %+.2f 11-12 %+.2f noisy %+.2f | dev 079 %.4f' % (d[:, (nz < 11) & ~f].mean(), d[:, ng & (nz >= 11)].mean(), d[:, g].mean(), N.rm(new)))
