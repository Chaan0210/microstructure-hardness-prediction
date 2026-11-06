"""Clean rows: does the hardness slope on het measurements attenuate when the grain count N is small (sampling noise in zone sizes)?
OLS y ~ controls + het + het*(300/N) + 300/N, standardised; numpy t-values."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float)
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[str(i)] for i in F['ids']])
k = lambda n: X[:, keys.index(n)]; nz = np.nan_to_num(k('noise') * 255, nan=0); c = (nz < 9.5) & (np.nan_to_num(k('n_obj')) >= 20)
def ols(Z, yy):
    A = np.c_[np.ones(len(Z)), Z]; b, *_ = np.linalg.lstsq(A, yy, rcond=None); r = yy - A @ b; s2 = (r @ r) / (len(yy) - A.shape[1]); se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * s2); return b, b / se
ctrl = ['u_ph', 'm_la_mean', 'm_el_pix', 'sh_cor15', 'z_f25', 'ang_c4', 'ang_s4']
def test(v, others):
    h = k(v)[c]; s = 300 / k('n_obj')[c]; Z = np.column_stack([k(n)[c] for n in others] + [h, (h - np.nanmean(h)) * s, s]); m = np.isfinite(Z).all(1)
    Zs = (Z[m] - Z[m].mean(0)) / Z[m].std(0); b, t = ols(Zs, y[c][m]); print(f'{v:18s} slope {b[-3]:+.2f} (t {t[-3]:+.1f})   x(300/N) interaction {b[-2]:+.2f} (t {t[-2]:+.1f})   n {m.sum()}')
for hv in ['het4_u', 'sizefield_std_s32', 'tile_logA_std', 'het4_u_mx']: test(hv, ctrl)
for v in ['m_la_mean', 'u_ph', 'z_f25']: test(v, [n for n in ctrl if n != v] + ['het4_u'])
