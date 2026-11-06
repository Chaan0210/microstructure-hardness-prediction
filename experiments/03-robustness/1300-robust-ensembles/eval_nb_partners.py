"""Evaluate nb (no het/intensity) plain partners on non-gated rows: dev (delta applied to production OOF) and holdout-100 (fit on dev-400)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, nsgain as N
from robust_eval import report
Z = np.load(_repo_io('nb-partners.npz')); y, g, c61, C, nz, fail = N.y, N.g, N.c61, N.C, N.nz, N.fail; f = fail[0]; ng = ~g & ~f
Cx = dict(C)
for nm, key in (('ridge', 'ridge'), ('logr', 'logridge'), ('gmm', 'gmm')): Cx[nm] = np.where(ng, C[nm] + (Z[key + '_nb_dev'] - Z[key + '_full_dev']), C[nm])
new = N.blend(Cx); d = (new - y) ** 2 - (c61 - y) ** 2
rng = np.random.RandomState(0); dm = d.mean(0); P = np.mean([dm[rng.randint(0, 400, 400)].mean() < 0 for _ in range(4000)])
print(f'dev: non-gated dMSE {d[:, ng].mean():+.2f} ± {d[:, ng].mean(0).std() / np.sqrt(ng.sum()):.2f}, clean<11 {d[:, (nz < 11) & ~f].mean():+.2f}, 11-12 {d[:, (nz >= 11) & ~g & ~f].mean():+.2f}; bootstrap P(better) {P:.3f}')
report('dev partners nb (non-gated)', new)
for k in ('ridge', 'logr', 'gmm'): Cy = dict(C); key = {'logr': 'logridge'}.get(k, k); Cy[k] = np.where(ng, C[k] + (Z[key + '_nb_dev'] - Z[key + '_full_dev']), C[k]); report(f'   only {k}', N.blend(Cy))
# holdout: proxy partner bundle (0.128 ridge + 0.128 logr + 0.15 gmm) and partner-alone errors on non-gated holdout rows
import csv
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; hid = [str(i) for i in Z['hid']]; yh = np.array([lab[i] for i in hid])
H = np.load(_repo_io('holdout-featdrop.npz')); assert list(H['hid']) == hid; nzh = H['noise']; ngh = nzh < 12
for key in ('ridge', 'logridge', 'gmm'):
    a, b = Z[key + '_full_ho'], Z[key + '_nb_ho']; dd = (b - yh)[ngh] ** 2 - (a - yh)[ngh] ** 2
    print(f'holdout {key:8s} non-gated RMSE full {np.sqrt(((a - yh)[ngh] ** 2).mean()):.3f} nb {np.sqrt(((b - yh)[ngh] ** 2).mean()):.3f}  dMSE {dd.mean():+.2f} ± {dd.std() / np.sqrt(ngh.sum()):.2f}')
S = ['038', '039', '044']; V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
tf = np.mean([H[f'{s}_{v}_full'] for s in S for v in V], 0); tn = np.mean([H[f'{s}_{v}_noboth'] for s in S for v in V], 0)
pf = 0.128 * Z['ridge_full_ho'] + 0.128 * Z['logridge_full_ho'] + 0.15 * Z['gmm_full_ho']; pn = 0.128 * Z['ridge_nb_ho'] + 0.128 * Z['logridge_nb_ho'] + 0.15 * Z['gmm_nb_ho']
for nm, b0, b1 in (('partners only', 0.384 * tf + pf, 0.384 * tf + pn), ('TabPFN only', 0.384 * tf + pf, 0.384 * tn + pf), ('both', 0.384 * tf + pf, 0.384 * tn + pn)):
    b0, b1 = b0 / 0.79, b1 / 0.79; dd = (b1 - yh)[ngh] ** 2 - (b0 - yh)[ngh] ** 2
    print(f'holdout proxy blend (tab+partners, non-gated n={ngh.sum()}): {nm:14s} dMSE {dd.mean():+.2f} ± {dd.std() / np.sqrt(ngh.sum()):.2f}  RMSE {np.sqrt(((b0 - yh)[ngh] ** 2).mean()):.3f} -> {np.sqrt(((b1 - yh)[ngh] ** 2).mean()):.3f}')
