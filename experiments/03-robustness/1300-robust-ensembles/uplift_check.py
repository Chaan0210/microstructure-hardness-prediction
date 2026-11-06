from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
exec(open(_repo_io('imgstats.py')).read().split("names = [")[0])
PS = np.load(_repo_io('pixstats.npz')); TX = PS['train_X']
for nm, m in [('clean <7', nz < 7), ('clean 7-9.5', (nz >= 7) & (nz < 9.5)), ('mid 9.5-12', (nz >= 9.5) & (nz < 12)), ('noisy 12-14', (nz >= 12) & (nz < 14)), ('noisy 17-21', nz >= 17)]:
    print(f'{nm:12s} n {m.sum():3d}  mid/hf {TX[m, 1].mean():.3f}  low/hf {TX[m, 2].mean():.3f}  grad p95 {TX[m, 6].mean():.2f}  blockmean p50-p10 {TX[m, 3].mean():.2f}  log hf {TX[m, 0].mean():.2f}')
# simulated noise-only uplift of real noisy sources (12-14.5) to 17-20 gray, measured with the same Haar estimator
from PIL import Image
rng = np.random.RandomState(0); out = []; meas = []
for i in np.where((nz >= 12) & (nz < 14.5))[0]:
    a = np.asarray(Image.open(_repo_io(f'{P}/data/train/{ids[i]}.png')).convert('L'), float); tgt = rng.uniform(17, 20); add = np.sqrt(max(tgt ** 2 - nz[i] ** 2, 1))
    b = np.clip(np.round(a + rng.randn(*a.shape) * add), 0, 255).astype(np.uint8); Image.fromarray(b).save(_repo_io('/tmp/_u.png')) if False else None
    hh = (b[0::2, 0::2].astype(float) - b[0::2, 1::2] - b[1::2, 0::2] + b[1::2, 1::2]) / 2; meas.append(np.median(np.abs(hh)) / 0.6745)
    Image.fromarray(b).save(_repo_io('_uplift_tmp.png')); out.append(stats('_uplift_tmp.png'))
out = np.array(out); print(f'uplifted 12-14.5 -> 17-20 (n {len(out)}): measured noise {np.mean(meas):.1f}  mid/hf {out[:, 1].mean():.3f}  low/hf {out[:, 2].mean():.3f}  grad p95 {out[:, 6].mean():.2f}  blockmean {out[:, 3].mean():.2f}  log hf {out[:, 0].mean():.2f}')
import os; os.remove('_uplift_tmp.png')
