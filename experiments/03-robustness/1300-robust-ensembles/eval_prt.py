from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, nsgain as N
w = np.load(_repo_io('lm_weights.npy')); y, g, C, c61 = N.y, N.g, N.C, N.c61
def lm(base, new):
    dm = (w * ((new - y) ** 2 - (base - y) ** 2))[:, g].mean(0); wr = lambda a: np.sqrt((w * (a - y) ** 2).mean()); return wr(base) - wr(new), dm.mean(), dm.std() / np.sqrt(g.sum())
R = np.load(_repo_io(f'prt-{sys.argv[1]}.npz')); f, gg, gc = R['f'], R['g'], R['gconst']
rn = lambda a: float(np.sqrt(((a - y)[:, g] ** 2).mean()))
print(f'noisy alone: TabPFN f {rn(f):.3f}  f-g {rn(f - gg):.3f}  f-gconst {rn(f - gc):.3f}  block {rn(C["tab"]):.3f}  block-g {rn(C["tab"] - gg):.3f}  logr(DTA) {rn(C["logr"]):.3f}')
print('corr(g, logr - block) on noisy rows', np.corrcoef(gg[:, g].ravel(), (C['tab'] - C['logr'])[:, g].ravel())[0, 1].round(3))
for nm, corr in [('block - g', gg), ('block - gconst', gc)]:
    Cx = dict(C); Cx['tab'] = np.where(g, C['tab'] - corr, C['tab']); new = N.blend(Cx); N.decomp(f'PRT {nm}', new); a, b, c = lm(c61, new); print(f'      LM-dev {a:+.4f} weighted noisy dMSE {b:+.2f} SE {c:.2f}')
