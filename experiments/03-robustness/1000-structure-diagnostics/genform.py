"""Reverse-engineering the label generator on CLEAN train images (pixel noise < 9.5; measurements are accurate there):
forward selection of OLS terms by repeated 10-fold CV RMSE over physically motivated candidates (+ squares / interactions of the selected ones).
usage: genform.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float); ids = [str(i) for i in F['ids']]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
P = np.load(_repo_io('/private/tmp/claude-501/-Users-dongchan-Desktop-PythonWorkspace-microstructure-2026/f62e347b-cb1a-4c17-8d54-2bf505be6706/scratchpad/presets.npz')); nz = dict(zip(P['tr'], P['ntr']))
cl = np.array([nz[i] < 9.5 for i in ids]); Xc = X[cl]; yc = y[cl]; print('clean rows', cl.sum(), ' y sd', yc.std().round(2))
cand = ['u_ph', 'm_ph_hard', 'z_f25', 'sh_cor15', 'aw_inv_sqrt_d', 'grain_inv_sqrt_d', 'aw_eqd_mean', 'grain_eqd_mean', 'n_obj', 'pore_frac', 'px_pore_frac', 'ar_wmean',
        'orient_coherence', 'ang_c4', 'ang_s4', 'grain_logA_std', 'het4_u', 'tile_spfrac_std', 'sp_contiguity', 'sp_to_matrix_size', 'sp_ar', 'm_el_pix', 'solidity_wmean',
        'coarse_area_frac', 'fine_area_frac', 'pore_d_mean', 'neighbors_mean', 'sp_z_mean', 'obj_I_cv', 'm_la_mean']
C = {k: Xc[:, keys.index(k)] for k in cand}
for k in list(C): C[k] = np.where(np.isfinite(C[k]), C[k], np.nanmedian(C[k]))
C['log_n_obj'] = np.log(C['n_obj']); C['inv_sqrt_eqd'] = 1 / np.sqrt(C['aw_eqd_mean']); C['log_eqd'] = np.log(C['aw_eqd_mean'])
def cv_rmse(cols, reps=10):
    Z = np.column_stack([C[c] for c in cols]) if cols else np.zeros((len(yc), 0)); e = []
    for s in range(reps):
        pr = np.zeros(len(yc))
        for tr, va in KFold(10, shuffle=True, random_state=s).split(yc):
            pr[va] = LinearRegression().fit(Z[tr], yc[tr]).predict(Z[va]) if cols else yc[tr].mean()
        e.append(np.sqrt(np.mean((pr - yc) ** 2)))
    return float(np.mean(e))
sel = []; cur = cv_rmse([])
print('intercept only', round(cur, 3))
for step in range(10):
    best = min(((cv_rmse(sel + [k]), k) for k in C if k not in sel), key=lambda t: t[0])
    if best[0] > cur - 0.02: break
    sel.append(best[1]); cur = best[0]; print(f'+ {best[1]:22s} CV RMSE {cur:.3f}')
Z = np.column_stack([C[c] for c in sel]); lr = LinearRegression().fit(Z, yc); print('coefs', dict(zip(sel, lr.coef_.round(3))), 'intercept', round(lr.intercept_, 2))
# interactions / squares among selected
for a in sel:
    C[a + '^2'] = C[a] ** 2
    for b in sel:
        if a < b: C[a + '*' + b] = C[a] * C[b]
for step in range(5):
    best = min(((cv_rmse(sel + [k]), k) for k in C if k not in sel), key=lambda t: t[0])
    if best[0] > cur - 0.02: break
    sel.append(best[1]); cur = best[0]; print(f'+ {best[1]:22s} CV RMSE {cur:.3f}')
np.save(_repo_io('genform_sel.npy'), np.array(sel))
