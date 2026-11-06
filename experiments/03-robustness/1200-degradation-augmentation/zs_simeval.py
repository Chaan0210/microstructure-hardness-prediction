"""Sim evaluation of ZS-net (cross-fitted): degraded copies (deg2/deg3) of clean train images; fold-A model predicts half-B images and vice versa.
Reports agreement with mask-based truth (het-true.npz) vs production het features, then the TabPFN floor CV (same 249 rows/labels) with the
deg2 feature set plus ZS features / with the het group replaced."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300')); sys.path.insert(0, '.')
from zs_feats import image_feats, KF
from floor_groups_def import grp
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
T = np.load(_repo_io('het-true.npz')); tid = [str(i) for i in T['ids']]; Zt = T['Z']; het_true = np.nanstd(Zt[:, :, 0], 1); phs_true = np.nanstd(Zt[:, :, 2], 1)
D = np.load(_repo_io('zs-prep.npz')); n = len(D['ids']); p = np.random.RandomState(0).permutation(n); A, B = set(p[:n // 2]), set(p[n // 2:])
pick = {tid[k]: (1 if k in A else 0) for k in range(n)}   # folds order in R = ['A', 'B']: images in A use model B (index 1)
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
y = np.array([lab[i] for i in tid]); XA = F['X'][[pos[i] for i in tid]].astype(float); G = np.array([grp(k) for k in keys])
run_cv = len(sys.argv) > 1 and sys.argv[1] == 'cv'
if run_cv:
    import harness
    from sklearn.model_selection import KFold
    def cv(X, reps=(0, 1)):
        out = []
        for r in reps:
            o = np.zeros(len(y))
            for tr, va in KFold(5, shuffle=True, random_state=r).split(X):
                m = harness.tabpfn_new('v3.5', r, 'mps'); m.fit(X[tr], y[tr]); o[va] = m.predict(X[va])
            out.append(o)
        return float(np.sqrt(((np.array(out) - y) ** 2).mean()))
for tag in ('deg2', 'deg3'):
    R = np.load(_repo_io(f'zs-synth-{tag}.npz')); assert [str(i) for i in R['ids']] == tid
    Z = image_feats(R['R'], tid, list(R['folds']), pick); S = np.load(_repo_io(f'../push-500/synthrows-train_{tag}-u1dmavgH2Z.npz'), allow_pickle=True); XB = S['X'].astype(float)
    print(f'== {tag}: agreement with mask truth (cross-fitted ZS-net)')
    print('  zs_het_b %.3f  zs_het_lam %.3f  | het4_u %.3f  sizefield_std_s32 %.3f  tile_logA_std %.3f' % tuple(
        [np.corrcoef(Z[:, KF.index(k)], het_true)[0, 1] for k in ('zs_het_b', 'zs_het_lam')] + [np.corrcoef(XB[:, keys.index(k)], het_true)[0, 1] for k in ('het4_u', 'sizefield_std_s32', 'tile_logA_std')]))
    print('  zs_ph_std vs true phase-zone std %.3f | tile_spfrac_std %.3f' % (np.corrcoef(Z[:, KF.index('zs_ph_std')], phs_true)[0, 1], np.corrcoef(np.nan_to_num(XB[:, keys.index('tile_spfrac_std')]), phs_true)[0, 1]))
    print('  y-corr: zs_het_b %.3f (true %.3f)  zs_ph_std %.3f (true %.3f)  zs_ph_mean %.3f' % (np.corrcoef(Z[:, 0], y)[0, 1], np.corrcoef(het_true, y)[0, 1], np.corrcoef(Z[:, 2], y)[0, 1], np.corrcoef(phs_true, y)[0, 1], np.corrcoef(Z[:, 6], y)[0, 1]))
    if run_cv and tag == 'deg2':
        print('  floor CV: deg2 baseline', round(cv(XB), 3), flush=True)
        print('  deg2 + ZS cols', round(cv(np.c_[XB, Z]), 3), flush=True)
        print('  deg2 with het group replaced by ZS cols', round(cv(np.c_[XB[:, G != 'het'], Z]), 3), flush=True)
        R0 = np.load(_repo_io('zs-clean.npz')) if False else None
