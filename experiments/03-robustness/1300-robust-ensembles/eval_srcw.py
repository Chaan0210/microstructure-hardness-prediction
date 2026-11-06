"""Evaluate srcw arms on the gated rows' DTA partners inside the 073 dev blend (non-gated rows = 073) and on the holdout-100 gated rows."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv, nsgain as N
from robust_eval import report
y, g, c61, C, fail = N.y, N.g, N.c61, N.C, N.fail
Z = np.load(_repo_io('srcw.npz')); c73 = np.load(_repo_io('dev-nb-candidate.npy'))
print('A reproduces production gated partners: ridge max|d| %.2e logridge %.2e' % (np.abs(Z['A_ridge'] - C['ridge'])[:, g].max(), np.abs(Z['A_logridge'] - C['logr'])[:, g].max()))
rb = lambda a, m: np.sqrt(((a - y) ** 2)[:, m].mean())
def blend(arm): return np.where(g, c73 + 0.128 * (Z[f'{arm}_ridge'] - C['ridge']) + 0.128 * (Z[f'{arm}_logridge'] - C['logr']), c73)
for arm in ('A', 'B', 'C'):
    print(f'{arm}: gated partner RMSE ridge {rb(Z[arm + "_ridge"], g):.3f} logridge {rb(Z[arm + "_logridge"], g):.3f}')
    report(f'   {arm} vs 073', blend(arm), base=c73)
report('   C vs B (pure weighting effect)', blend('C'), base=blend('B'))
H = np.load(_repo_io('srcw-ho.npz')); lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; hid = [str(i) for i in H['hid']]; yh = np.array([lab[i] for i in hid])
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); k = [str(x) for x in F['keys']]; pos = {str(i): j for j, i in enumerate(F['ids'])}
gh = np.nan_to_num(F['X'][[pos[i] for i in hid], k.index('noise')].astype(float) * 255, nan=0) >= 12
for arm in ('A', 'B', 'C'):
    pr = (H[f'{arm}_ridge'] + H[f'{arm}_logridge']) / 2; base = (H['A_ridge'] + H['A_logridge']) / 2; d = (pr - yh)[gh] ** 2 - (base - yh)[gh] ** 2
    print(f'holdout gated (n={gh.sum()}) {arm}: ridge {np.sqrt(((H[arm + "_ridge"] - yh)[gh] ** 2).mean()):.3f} logridge {np.sqrt(((H[arm + "_logridge"] - yh)[gh] ** 2).mean()):.3f} | partner-mean dMSE vs A {d.mean():+.2f} ± {d.std() / np.sqrt(gh.sum()):.2f}')
