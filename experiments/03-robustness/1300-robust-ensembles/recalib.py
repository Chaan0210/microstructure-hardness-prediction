"""Label-free re-calibration of the clean->noisy degradation (push-900/realdeg2.degrade2) on pixel statistics: grid over boundary removal
fraction, extra blur and dark-grain contrast; 60 clean train sources, noise drawn from the real >= 12 pool; distance to the real noisy train images
on 5 standardised statistics (log mid/hf, log low/hf, grad p95, blockmean p50-p10, p90-p50)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, itertools
from PIL import Image
sys.path.insert(0, _repo_io('../push-900')); from realdeg2 import degrade2
exec(open(_repo_io('imgstats.py')).read().split("names = [")[0])
M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'))['masks']; PS = np.load(_repo_io('pixstats.npz')); TX = PS['train_X']
J = [1, 2, 6, 3, 4]; real = TX[nz >= 12][:, J]; mu, sd = real.mean(0), real.std(0); pool = np.sort(nz[nz >= 12])
src = np.where(nz < 9.5)[0][::4][:60]
def sim(prm, seed=0):
    out = []
    for k in src:
        r = np.random.RandomState(seed * 1000 + k); a = np.asarray(Image.open(_repo_io(f'{P}/data/train/{ids[k]}.png')).convert('L'), float)
        tgt = float(pool[r.randint(len(pool))]); add = float(np.sqrt(max(tgt ** 2 - (0.4 * nz[k]) ** 2, 1.0)))
        g = degrade2(a, M[k], r, prm, noise=add).astype(np.uint8); Image.fromarray(g).save(_repo_io('_c.png')); out.append(np.array(stats('_c.png'))[J])
    return np.array(out)
res = []
for (plo, phi), blur, mud in itertools.product([(0.28, 0.68), (0.6, 0.9), (0.9, 1.0)], [0.75, 1.05, 1.35], [0.675, 0.55]):
    S = sim(dict(P_LO=plo, P_HI=phi, BLUR_MU=blur, MU_D=mud)); dist = np.sqrt((((S.mean(0) - mu) / sd) ** 2).sum())
    res.append((dist, plo, phi, blur, mud, S.mean(0))); print(f'P {plo:.2f}-{phi:.2f} blur {blur:.2f} mu_d {mud:.3f}: dist {dist:.3f}  stats {np.round(S.mean(0), 3)}', flush=True)
print('real', np.round(mu, 3)); best = min(res); print('BEST', best[:5])
import os; os.remove('_c.png')
