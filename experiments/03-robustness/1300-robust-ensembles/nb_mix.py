"""Feature-set ensemble on non-gated rows: (1-a)*full + a*nb for the TabPFN block and the three plain partners (dev + holdout). Train-only:
dev a-grid, cross-fitted a (chosen on 4 folds of each partition, applied to the 5th), holdout-100 proxy for a = 0.5 and 1."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, csv
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import align_dev
import nsgain as N
y, g, c61, C, nz, fail, fold = N.y, N.g, N.c61, N.C, N.nz, N.fail, N.fold; f = fail[0]; ng = ~g & ~f
S = ['038', '039', '044']; V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']; Tn = np.mean([align_dev(f'abl/dev-nb-{s}-{v}.npz') for s in S for v in V], 0); Z = np.load(_repo_io('nb-partners.npz'))
def build(a):
    Cx = dict(C); Cx['tab'] = np.where(ng, C['tab'] + a * (Tn - C['tab']), C['tab'])
    for nm, key in (('ridge', 'ridge'), ('logr', 'logridge'), ('gmm', 'gmm')): Cx[nm] = np.where(ng, C[nm] + a * (Z[key + '_nb_dev'] - Z[key + '_full_dev']), C[nm])
    return N.blend(Cx)
for a in (0.25, 0.5, 0.75, 1.0):
    b = build(a); d = (b - y) ** 2 - (c61 - y) ** 2; pp = np.sqrt(((c61 - y) ** 2).mean(1)) - np.sqrt(((b - y) ** 2).mean(1))
    print(f'a {a:.2f}: dev gain {N.rm(c61) - N.rm(b):+.4f} ({"/".join(f"{x:+.3f}" for x in pp)}) non-gated dMSE {d[:, ng].mean():+.2f} ± {d[:, ng].mean(0).std() / np.sqrt(ng.sum()):.2f}')
grid = np.linspace(0, 1.5, 31); B = {a: build(a) for a in grid}; cf = c61.copy(); ch = []
for k in range(3):
    for fo in range(5):
        tr = (fold[k] != fo) & ng; te = fold[k] == fo
        a = grid[np.argmin([((B[a][k] - y) ** 2)[tr].mean() for a in grid])]; ch.append(a); cf[k, te] = B[a][k, te]
print('cross-fitted a: mean %.2f sd %.2f min %.2f max %.2f | gain %+.4f' % (np.mean(ch), np.std(ch), np.min(ch), np.max(ch), N.rm(c61) - N.rm(cf)))
# holdout proxy
H = np.load(_repo_io('holdout-featdrop.npz')); lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
hid = [str(i) for i in H['hid']]; yh = np.array([lab[i] for i in hid]); m = H['noise'] < 12; assert [str(i) for i in Z['hid']] == hid
tf = np.mean([H[f'{s}_{v}_full'] for s in S for v in V], 0); tn = np.mean([H[f'{s}_{v}_noboth'] for s in S for v in V], 0)
pf = {k: Z[f'{k}_full_ho'] for k in ('ridge', 'logridge', 'gmm')}; pn = {k: Z[f'{k}_nb_ho'] for k in ('ridge', 'logridge', 'gmm')}
def hb(a): return (0.384 * (tf + a * (tn - tf)) + sum(w * (pf[k] + a * (pn[k] - pf[k])) for k, w in (('ridge', 0.128), ('logridge', 0.128), ('gmm', 0.15)))) / 0.79
for a in (0.5, 1.0):
    d = (hb(a) - yh)[m] ** 2 - (hb(0) - yh)[m] ** 2; print(f'holdout a {a}: non-gated dMSE {d.mean():+.2f} ± {d.std() / np.sqrt(m.sum()):.2f}')
