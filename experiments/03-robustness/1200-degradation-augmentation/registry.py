"""Registry of dev-evaluated variants (every slot variant with all 4 U-Net-variant dev files in push-300..1200, and every partner variant sr-*.npz),
each scored inside the exact 053 structure (slot variants: replace their family slot 038/039/044 on all rows; partner variants: RidgeCV/logRidge
replaced on noise >= 12 rows). Stores the per-row paired dSE (400 rows, partitions averaged) -> registry.npz. usage: registry.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, glob, re, numpy as np
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
exec(open(_repo_io('eval053.py')).read().split("base = {s: slot(L(f))")[0].split('"""', 2)[2])
nz = np.nan_to_num(np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['noise'], nan=0); m12 = (nz >= 12)[None] & ~B['fail']
base = {s: slot(L(f)) for s, f in BASE.items()}
def blend(sl, r=None, l=None):
    g = np.mean([sl[s] for s in ('38', '39', '44')], 0); c = 0.16 * B['p021'] + 0.384 * g + 0.128 * (B['ridge'] if r is None else r) + 0.128 * (B['logridge'] if l is None else l) + 0.05 * B['set'] + 0.15 * B['gmm']
    return np.where(B['fail'], B['p021'], c)
c0 = blend(base); e0 = (c0 - y) ** 2; print('053 dev', np.sqrt(e0.mean()).round(4))
names, R = [], []
for f in sorted(glob.glob('../push-*/dev-u1dmavgH2Z-*.npz')):
    d, fn = f.rsplit('/', 1); tag = fn[len('dev-u1dmavgH2Z-'):-4]; fs = [f'{d}/dev-{v}-{tag}.npz' for v in V]
    if not all(__import__('os').path.exists(x) for x in fs): continue
    try: s = slot(L(fs))
    except Exception as ex: continue
    fam = '44' if 'anghcor' in tag else ('39' if 'angh' in tag else '38'); sl = dict(base); sl[fam] = s
    r = (((blend(sl) - y) ** 2) - e0).mean(0)
    if np.abs(r).max() == 0: continue
    names.append(f'slot{fam}:{d[3:]}:{tag}'); R.append(r)
for f in sorted(glob.glob('sr-*.npz')):
    S = np.load(_repo_io(f))
    if S['ridge'].shape != (3, 400): continue
    r = (((blend(base, np.where(m12, S['ridge'], B['ridge']), np.where(m12, S['logridge'], B['logridge'])) - y) ** 2) - e0).mean(0)
    if np.abs(r).max() == 0: continue
    names.append('partner:' + f[3:-4]); R.append(r)
R = np.array(R); z = R.mean(1) / (R.std(1) / 20)
np.savez(_repo_io('registry.npz'), names=np.array(names), R=R); print('variants', len(names))
o = np.argsort(R.mean(1))
for i in list(o[:12]) + list(o[-5:]): print(f'{names[i]:55s} dMSE {R[i].mean():+.3f} z {z[i]:+.2f}')
