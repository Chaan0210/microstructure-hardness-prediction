"""053 structure, noisy rows (pixel noise >= 12): sim-augmented Ridge/log-Ridge (sr-A-<RT>.npz) and optionally gmm (sg-<GT>.npz) replace the
production partners; the augmented partners' share on noisy rows is scaled by K (rest of 053 proportionally). Also the stretch-free version
(change regressed on the per-partition centred 053 noisy prediction, that component removed). usage: eval_aug_all.py RT GT|none K [K ...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
B = load(3); y = B['y']; fail = B['fail']; nz = np.nan_to_num(np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['noise'], nan=0); gate = (nz >= 12) & ~fail[0]
S = np.load(_repo_io(f'sr-A-{sys.argv[1]}.npz')); gm = B['gmm'] if sys.argv[2] == 'none' else np.load(_repo_io(f'sg-{sys.argv[2]}.npz'))['oof']
c53 = np.where(fail, B['p021'], 0.16 * B['p021'] + 0.384 * B['g'] + 0.128 * B['ridge'] + 0.128 * B['logridge'] + 0.05 * B['set'] + 0.15 * B['gmm'])
aug_aw = 0.256 + (0.15 if sys.argv[2] != 'none' else 0.0)
other = (c53 - 0.128 * B['ridge'] - 0.128 * B['logridge'] - (0.15 * B['gmm'] if sys.argv[2] != 'none' else 0)) / (1 - aug_aw)
part = (0.128 * S['ridge'] + 0.128 * S['logridge'] + (0.15 * gm if sys.argv[2] != 'none' else 0)) / aug_aw
def rep(nm, a):
    e = a - y; print(f'{nm:30s} dev {np.sqrt((e ** 2).mean()):.4f} parts {np.round(np.sqrt((e ** 2).mean(1)), 4)} noisy {np.sqrt((e[:, gate] ** 2).mean()):.3f} sd {a[:, gate].std(1).mean():.2f}')
rep('053', c53)
for K in map(float, sys.argv[3:]):
    w = min(aug_aw * K, 1.0); a = np.where(gate[None], (1 - w) * other + w * part, c53); rep(f'K={K:g} (aug share {w:.3f})', a)
    o = a.copy()
    for p in range(3):
        x = c53[p, gate] - c53[p, gate].mean(); d = (a - c53)[p, gate]; o[p, gate] = c53[p, gate] + d - (x * d).sum() / (x * x).sum() * x
    rep(f'  stretch-free', o)
