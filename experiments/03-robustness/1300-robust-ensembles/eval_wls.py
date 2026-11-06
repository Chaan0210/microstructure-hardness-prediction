from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, nsgain as N
from blend_dev import load
B = load(3); y, g, C, c61, nz, fail = N.y, N.g, N.C, N.c61, N.nz, N.fail; lw = np.load(_repo_io('lm_weights.npy')); W = np.load(_repo_io(f'wls-p{sys.argv[1]}.npz'))
ng = ~g & ~fail[0]; rm = lambda a, m: float(np.sqrt(((a - y)[:, m] ** 2).mean()))
print(f'partners alone: clean/mid ridge {rm(B["ridge"], ng):.3f} -> WLS {rm(W["ridge_0"], ng):.3f}; logr {rm(B["logridge"], ng):.3f} -> {rm(W["logridge_0"], ng):.3f} | noisy DTA ridge {rm(C["ridge"], g):.3f} -> {rm(W["ridge_1"], g):.3f}; logr {rm(C["logr"], g):.3f} -> {rm(W["logridge_1"], g):.3f}')
def show(nm, Cx):
    a = N.blend(Cx); d = ((a - y) ** 2 - (c61 - y) ** 2); dn = d[:, ng].mean(0); dg = (lw * d)[:, g].mean(0); info, _ = N.decomp('', a, quiet=True)
    print(f'{nm:30s} dev {N.rm(a):.4f} ({N.rm(c61) - N.rm(a):+.4f}; parts {"/".join(f"{v:+.3f}" for v in np.sqrt(((c61 - y) ** 2).mean(1)) - np.sqrt(((a - y) ** 2).mean(1)))}) | clean+mid dMSE {dn.mean():+.2f} (SE {dn.std() / np.sqrt(ng.sum()):.2f}) | noisy non-stretch {info:+.4f} LM dMSE {dg.mean():+.2f} (SE {dg.std() / np.sqrt(g.sum()):.2f})')
Cx = dict(C); Cx['ridge'] = np.where(g, C['ridge'], W['ridge_0']); Cx['logr'] = np.where(g, C['logr'], W['logridge_0']); show('WLS on clean/mid partners', Cx)
Cy = dict(C); Cy['ridge'] = np.where(g, W['ridge_1'], C['ridge']); Cy['logr'] = np.where(g, W['logridge_1'], C['logr']); show('WLS on noisy DTA partners', Cy)
Cz = dict(Cx); Cz['ridge'] = np.where(g, W['ridge_1'], Cx['ridge']); Cz['logr'] = np.where(g, W['logridge_1'], Cx['logr']); show('WLS both', Cz)
