"""TabPFN feature-group ablation (044 slot, u1dmavgH2Z, seeds 0,1,2) -> member RMSE per band vs full, and the blend effect if the same change
were applied to every block member (block + (member_drop - member_full)); noisy rows also slope-free."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import io_path as _repo_io
import sys, glob, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import align_dev
import nsgain as N
y, c61, C, nz, fail = N.y, N.c61, N.C, N.nz, N.fail; f = fail[0]
bands = {'clean<11': (nz < 11) & ~f, '11-12': (nz >= 11) & (nz < 12) & ~f, 'noisy>=12': (nz >= 12) & ~f}
base = align_dev('abl/dev-abl-none-s012.npz'); rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
print('full member: ' + ' '.join(f'{k} {rb(base, m):.3f}' for k, m in bands.items()))
for fn in sorted(glob.glob('abl/dev-abl-*-s012.npz')):
    gname = fn.split('dev-abl-')[1].rsplit('-s012', 1)[0]
    if gname == 'none': continue
    a = align_dev(fn); Cx = dict(C); Cx['tab'] = C['tab'] + (a - base); new = N.blend(Cx); d = (new - y) ** 2 - (c61 - y) ** 2
    pp = np.sqrt(((c61 - y) ** 2).mean(1)) - np.sqrt(((new - y) ** 2).mean(1))
    print(f'drop {gname:9s} member ' + ' '.join(f'{rb(a, m) - rb(base, m):+.3f}' for m in bands.values()) +
          f' | blend(all members) gain {N.rm(c61) - N.rm(new):+.4f} ({"/".join(f"{x:+.3f}" for x in pp)}) dMSE ' + ' '.join(f'{k} {d[:, m].mean():+.2f}({d[:, m].mean(0).std() / np.sqrt(m.sum()):.2f})' for k, m in bands.items()))
