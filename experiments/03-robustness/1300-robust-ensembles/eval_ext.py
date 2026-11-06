"""Extension screens on top of the 078 dev blend (076 dev + H9 columns in plain partners): replace the 044 slot's H9 members (nb+h9 on non-gated rows,
full+h9 on gated rows) by h9s (+ sigma 2/4 p34, contrast), h9r (+4 random columns) or h9g (+ GMM phase_fraction, log_contrast). Other 8 members unchanged."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import align_dev
import nsgain as N
from robust_eval import report
y, g, fail, nz = N.y, N.g, N.fail, N.nz; f = fail[0]; ng = ~g & ~f; V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
Z = np.load(_repo_io('h9-partners.npz')); c76 = np.load(_repo_io('dev-combo.npy'))
c78 = c76 + np.where(ng, sum(w * (Z[f'phys_{k}'] - Z[f'none_{k}']) for k, w in (('ridge', 0.128), ('logridge', 0.128), ('gmm', 0.15))), 0); np.save(_repo_io('dev-078.npy'), c78)
print('078 dev %.4f (076 %.4f, 061 %.4f)' % (N.rm(c78), N.rm(c76), N.rm(N.c61)))
h9 = np.mean([np.where(ng, align_dev(f'abl/dev-h9phys-nb-044-{v}.npz'), align_dev(f'abl/dev-h9phys-full-044-{v}.npz')) for v in V], 0)
for tag in sys.argv[1:]:
    try: new = np.mean([np.where(ng, align_dev(f'abl/dev-{tag}-nb-044-{v}.npz'), align_dev(f'abl/dev-{tag}-full-044-{v}.npz')) for v in V], 0)
    except FileNotFoundError: print(tag, 'not complete'); continue
    d = 0.384 * (new - h9) / 3   # the 044 slot is one third of the TabPFN block
    cand = np.where(fail, c78, c78 + d); report(f'078 + 044-slot {tag}', cand, base=c78)
    dd = (cand - y) ** 2 - (c78 - y) ** 2; print(f'      bands: clean<11 {dd[:, (nz < 11) & ~f].mean():+.2f} 11-12 {dd[:, ng & (nz >= 11)].mean():+.2f} noisy {dd[:, g].mean():+.2f}')
