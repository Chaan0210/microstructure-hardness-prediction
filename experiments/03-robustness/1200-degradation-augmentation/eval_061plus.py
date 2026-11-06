"""061 structure (Ridge/logRidge replaced on noise>=12 rows by sr-A-d1234n12) plus optional same-weight gmm replacement (sg-<tag>)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
B = load(3); y = B['y']; fail = B['fail']; nz = np.nan_to_num(np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['noise'], nan=0); gate = (nz >= 12) & ~fail[0]
S = np.load(_repo_io('sr-A-d1234n12.npz'))
c53 = np.where(fail, B['p021'], 0.16 * B['p021'] + 0.384 * B['g'] + 0.128 * B['ridge'] + 0.128 * B['logridge'] + 0.05 * B['set'] + 0.15 * B['gmm'])
c61 = np.where(gate[None], c53 + 0.128 * (S['ridge'] - B['ridge']) + 0.128 * (S['logridge'] - B['logridge']), c53)
r = lambda a: f'{np.sqrt(((a - y) ** 2).mean()):.4f} parts {np.round(np.sqrt(((a - y) ** 2).mean(1)), 4)} noisy {np.sqrt(((a - y)[:, gate] ** 2).mean()):.3f}'
print('053', r(c53)); print('061', r(c61))
for t in sys.argv[1:]:
    G = np.load(_repo_io(f'sg-{t}.npz'))['oof']; print(f'061 + gmm {t}', r(np.where(gate[None], c61 + 0.15 * (G - B['gmm']), c61)))
