from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from PIL import Image
exec(open(_repo_io('imgstats.py')).read().split("names = [")[0])
def uplift(src_mask, lo, hi, seed=0):
    rng = np.random.RandomState(seed); out, meas = [], []
    for i in np.where(src_mask)[0]:
        a = np.asarray(Image.open(_repo_io(f'{P}/data/train/{ids[i]}.png')).convert('L'), float); tgt = rng.uniform(lo, hi); add = np.sqrt(max(tgt ** 2 - nz[i] ** 2, 1))
        b = np.clip(np.round(a + rng.randn(*a.shape) * add), 0, 255).astype(np.uint8); Image.fromarray(b).save(_repo_io('_u.png')); out.append(stats('_u.png'))
        hh = (b[0::2, 0::2].astype(float) - b[0::2, 1::2] - b[1::2, 0::2] + b[1::2, 1::2]) / 2; meas.append(np.median(np.abs(hh)) / 0.6745)
    return np.array(out), np.mean(meas)
PS = np.load(_repo_io('pixstats.npz')); TX = PS['train_X']; ref = (nz >= 13) & (nz < 16)
show = lambda nm, S, mn: print(f'{nm:30s} noise {mn:5.1f}  mid/hf {S[:, 1].mean():.3f}  low/hf {S[:, 2].mean():.3f}  grad p95 {S[:, 6].mean():.2f}  blockmean {S[:, 3].mean():.2f}  hf {S[:, 0].mean():.2f}')
show('real 13-16', TX[ref], nz[ref].mean())
S, mn = uplift((nz >= 9.5) & (nz < 12), 13, 16); show('mid 9.5-12 uplifted to 13-16', S, mn)
S, mn = uplift(nz < 9.5, 13, 16); show('clean uplifted to 13-16 (noise only)', S, mn)
import os; os.remove('_u.png')
