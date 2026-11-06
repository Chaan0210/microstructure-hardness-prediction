"""Blend-level independent check on the 100 train rows outside the Study900 dev-400 (never used for selection): every component refitted on the 400
dev rows and predicted on the holdout. Proxy 053 = (0.16 CNN + 0.384 TabPFN slots + 0.128 RidgeCV + 0.128 logRidge + 0.15 gmm) / 0.95 (set NN
omitted, CNN = TinyCNN128 ctrl ensemble as stand-in for 021). Candidates change components on noisy rows exactly as on test.
usage: holdout_blend.py  (needs holdout-ta.npz, holdout-cnn-*.npz)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, glob, csv, numpy as np, warnings
warnings.filterwarnings('ignore')
exec(open(_repo_io('holdout_sets.py')).read().split('for name, sets in')[0].split('"""', 2)[2])
E = '..'; import importlib
src = open(_repo_io('../push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]; gp = {'__file__': '../push-900/gmm_partner.py', '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), gp)
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; tr = np.array([i in dev for i in ids]); hid = list(np.array(ids)[~tr])
Btr = np.load(_repo_io('../probabilistic-grains-900/object-bank-train.npz'), allow_pickle=True); XG = np.c_[F['X'].astype(float), gp['gmm_feats'](Btr, ids)]; yy = np.array([lab[i] for i in ids])
gmm, _ = gp['fit_predict'](XG[tr], yy[tr], XG[~tr], 19001)
def partners(sets, wide):
    pr, y, nz = run('u1dmavgH2Z', sets, wide, 'ridge'); pl = np.exp(np.mean([np.log(run(v, sets, wide, 'log')[0]) for v in V], 0)); return pr, pl, y, nz
r0, l0, y, nz = partners([], False); r61, l61, _, _ = partners('deg,deg2,deg3,deg4'.split(','), False); r62, l62, _, _ = partners('deg,deg2,deg3,deg4,deg5,deg6,deg7'.split(','), True)
T = np.load(_repo_io('holdout-ta.npz'), allow_pickle=True); assert [str(i) for i in T['ids']] == hid
sp = np.mean([T[f'{s}_plain'] for s in ('038', '039', '044')], 0); sta = np.mean([T[f'{s}_TA'] for s in ('038', '039', '044')], 0)
cid = [str(i) for i in np.load(_repo_io(sorted(glob.glob('holdout-cnn-ctrl-s*.npz'))[0]))['ids']]; jj = [cid.index(i) for i in hid]
seeds = sorted(int(f.split('-s')[-1][:-4]) for f in glob.glob('holdout-cnn-dta-s*.npz'))
cc = np.mean([np.load(_repo_io(f'holdout-cnn-ctrl-s{s}.npz'))['pred'] for s in seeds], 0)[jj]; cd = np.mean([np.load(_repo_io(f'holdout-cnn-dta-s{s}.npz'))['pred'] for s in seeds], 0)[jj]
g = nz >= 12; G = g.astype(float)
blend = lambda c, s, r, l: (0.16 * c + 0.384 * s + 0.128 * r + 0.128 * l + 0.15 * gmm) / 0.95
P = {'053': blend(cc, sp, r0, l0)}
P['061'] = blend(cc, sp, np.where(g, r61, r0), np.where(g, l61, l0)); P['062'] = blend(cc, sp, np.where(g, r62, r0), np.where(g, l62, l0))
P['064'] = blend(cc, np.where(g, sta, sp), np.where(g, r62, r0), np.where(g, l62, l0)); P['061+cnn'] = blend(np.where(g, cd, cc), sp, np.where(g, r61, r0), np.where(g, l61, l0))
P['061+TA'] = blend(cc, np.where(g, sta, sp), np.where(g, r61, r0), np.where(g, l61, l0))
print(f'holdout n={len(y)} noisy {g.sum()}; CNN seeds {len(seeds)}; components noisy RMSE: cnn ctrl {np.sqrt(((cc - y)[g] ** 2).mean()):.2f} dta {np.sqrt(((cd - y)[g] ** 2).mean()):.2f} slots {np.sqrt(((sp - y)[g] ** 2).mean()):.2f} TA {np.sqrt(((sta - y)[g] ** 2).mean()):.2f} gmm {np.sqrt(((gmm - y)[g] ** 2).mean()):.2f}')
e61 = (P['061'] - y) ** 2
for k, p in P.items():
    e = (p - y) ** 2; d = (e - e61)[g]
    print(f'{k:8s} all {np.sqrt(e.mean()):.3f} noisy {np.sqrt(e[g].mean()):.3f} | vs 061 noisy dMSE {d.mean():+.2f} (SE {d.std() / np.sqrt(g.sum()):.2f})')
