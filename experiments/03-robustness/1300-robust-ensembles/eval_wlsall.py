"""061 with WLS partners (Ridge/logRidge plain on clean+mid rows, DTA on noisy rows, gmm-Ridge everywhere): plain dev, partitions, clean+mid dMSE,
noisy non-stretch and LM criteria, bootstrap P."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, nsgain as N
y, g, C, c61, nz, fail = N.y, N.g, N.C, N.c61, N.nz, N.fail; lw = np.load(_repo_io('lm_weights.npy')); W = np.load(_repo_io('wls-p1.0.npz')); Gm = np.load(_repo_io('wls-gmm.npz'))['oof']
ng = ~g & ~fail[0]
def show(nm, Cx):
    a = N.blend(Cx); d = (a - y) ** 2 - (c61 - y) ** 2; dn = d[:, ng].mean(0); dg = (lw * d)[:, g].mean(0); info, _ = N.decomp('', a, quiet=True)
    allr = d.mean(0); rng = np.random.RandomState(0); bs = [allr[rng.randint(0, 400, 400)].mean() for _ in range(4000)]
    print(f'{nm:26s} dev {N.rm(a):.4f} ({N.rm(c61) - N.rm(a):+.4f}; parts {"/".join(f"{v:+.3f}" for v in np.sqrt(((c61 - y) ** 2).mean(1)) - np.sqrt(((a - y) ** 2).mean(1)))}) clean+mid dMSE {dn.mean():+.2f} (SE {dn.std() / np.sqrt(ng.sum()):.2f}) | noisy non-stretch {info:+.4f} LM {dg.mean():+.2f} | all-row P(better) {np.mean(np.array(bs) < 0):.2f}')
Cg = dict(C); Cg['gmm'] = Gm; show('WLS gmm only', Cg)
Cx = dict(C); Cx['ridge'] = np.where(g, W['ridge_1'], W['ridge_0']); Cx['logr'] = np.where(g, W['logridge_1'], W['logridge_0']); show('WLS Ridge/logRidge', Cx)
Cx['gmm'] = Gm; show('WLS all three partners', Cx)
