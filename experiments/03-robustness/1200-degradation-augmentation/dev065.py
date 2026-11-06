"""Build the 065 dev blend (053 structure; partners sr-A-d1to7n12-wal on noise>=12, three TabPFN slots + TA column on noise>=9.5) and save
all components (Study900 order, 3 partitions x 400) to dev065.npz for decomposition. usage: dev065.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
exec(open(_repo_io('eval053.py')).read().split("base = {s: slot(L(f))")[0].split('"""', 2)[2])
nz = np.nan_to_num(np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['noise'], nan=0); fail = B['fail'][0]
base = {s: slot(L(f)) for s, f in BASE.items()}
TA = {'38': 'dev-{v}-v35TA.npz', '39': 'dev-{v}-anghTA.npz', '44': 'dev-{v}-anghcorTA.npz'}
ta = {s: slot(L([t.format(v=v) for v in V])) for s, t in TA.items()}
g12, g95 = (nz >= 12) & ~fail, (nz >= 9.5) & ~fail
S = np.load(_repo_io('sr-A-d1to7n12-wal.npz'))
slots = {s: np.where(g95[None], ta[s], base[s]) for s in base}
g = np.mean([slots[s] for s in ('38', '39', '44')], 0)
r = np.where(g12[None], S['ridge'], B['ridge']); l = np.where(g12[None], S['logridge'], B['logridge'])
c = 0.16 * B['p021'] + 0.384 * g + 0.128 * r + 0.128 * l + 0.05 * B['set'] + 0.15 * B['gmm']; c = np.where(B['fail'], B['p021'], c)
print('065 dev', np.sqrt(((c - y) ** 2).mean()).round(4))
np.savez(_repo_io('dev065.npz'), c=c, y=y, nz=nz, fail=fail, p021=B['p021'], g=g, ridge=r, logridge=l, set=B['set'], gmm=B['gmm'], ids=np.array(sid),
         **{f's{s}': slots[s] for s in slots}, **{f'b{s}': base[s] for s in base})
bands = [('clean', nz < 9.5), ('mid', (nz >= 9.5) & (nz < 12)), ('noisy', g12)]
print('component RMSE by band (mean over partitions)')
for k in ['c', 'p021', 'g', 'ridge', 'logridge', 'set', 'gmm']:
    a = c if k == 'c' else (g if k == 'g' else (r if k == 'ridge' else (l if k == 'logridge' else B[k])))
    print(f'{k:9s} all {np.sqrt(((a - y) ** 2).mean()):.3f} ' + ' '.join(f'{b} {np.sqrt(((a - y)[:, m & ~fail] ** 2).mean()):.3f}' for b, m in bands))
# leave-one-component-out on noisy rows (renormalise remaining weights) and residual correlation with the blend
W = dict(p021=0.16, g=0.384, ridge=0.128, logridge=0.128, set=0.05, gmm=0.15); comp = dict(p021=B['p021'], g=g, ridge=r, logridge=l, set=B['set'], gmm=B['gmm'])
m = g12
for k in W:
    cc = sum(W[j] * comp[j] for j in W if j != k) / (1 - W[k]); cc = np.where(m[None], cc, c)
    e = (comp[k] - y)[:, m]; ec = (c - y)[:, m]
    print(f'drop {k:9s} on noisy -> all {np.sqrt(((cc - y) ** 2).mean()):.4f}  noisy {np.sqrt(((cc - y)[:, m] ** 2).mean()):.3f}  resid corr w/ blend {np.corrcoef(e.ravel(), ec.ravel())[0, 1]:.3f}')
