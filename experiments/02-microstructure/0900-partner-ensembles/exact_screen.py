"""Does the label use an exact per-grain aggregation that our summary features miss? Clean dev rows (noise < 9.5) only, zero-shot grain objects
(probabilistic-grains-900/object-bank-train.npz). Candidate aggregates: Hall-Petch-type terms with several size definitions (equivalent diameter,
bbox max, minor axis from AR) and exponents (-1/2, -1), number- vs area-weighted, with/without border grains, phase-specific versions, phase x size
cross terms. Screen against the 051 dev residual (mean over the 3 partitions): Spearman partial on the prediction, split halves A/B; plus the 1/N law
of the clean residual. usage: exact_screen.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from pathlib import Path
from scipy import stats
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
D = np.load(_repo_path(E, 'probabilistic-grains-900/ridge/predictions.npz'), allow_pickle=True); ids = [str(i) for i in D['ids']]; y, p, g, f, r = D['y'], D['p021'], D['grain045'], D['failed'], D['ridge']
L = np.load(_repo_path(H, 'diverse-ridge_a2_logy_var4.npz'))['oof']; P = np.where(f, p, 0.2 * p + 0.48 * g + 0.16 * r + 0.16 * L)
res = (y - P).mean(0); pred = P.mean(0); yy = y[0]
nz = dict(zip([str(i) for i in np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True)['ids']], np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255))
clean = np.array([nz[i] < 9.5 for i in ids])
B = np.load(_repo_path(E, 'probabilistic-grains-900/object-bank-train.npz'), allow_pickle=True); bid = [str(i) for i in B['ids']]; ok_ = B['object_keys'].tolist(); O = B['objects']; off = B['offsets']
col = lambda k: ok_.index(k)
def feats(i):
    j = bid.index(i); o = O[off[j]:off[j + 1]]; a = o[:, col('area')]; dark = o[:, col('shade_darkness')]; pore = o[:, col('pore_flag')] > 0.5; bd = o[:, col('border')] > 0.5
    ar = np.exp(o[:, col('log_ar')]); bb = o[:, col('bbox_max')]; d = np.sqrt(4 * a / np.pi); dmin = d / np.sqrt(ar); c4 = o[:, col('cos2')] ** 2 - o[:, col('sin2')] ** 2
    m = ~pore & (a >= 4); ph = m & (dark > 0.15); mx = m & ~ph; F = {}
    A = a[m].sum(); F['phase_area'] = a[ph].sum() / A; F['phase_count'] = ph.sum() / max(m.sum(), 1); F['pore_area'] = a[pore].sum() / 65536
    for nm, s in (('d', d), ('bb', bb), ('dmin', dmin)):
        for ex in (0.5, 1.0):
            v = s ** -ex
            for sel, tag in ((m, 'all'), (m & ~bd, 'int'), (ph, 'ph'), (mx, 'mx')):
                if sel.sum() < 2: F[f'{nm}{ex}_{tag}_n'] = F[f'{nm}{ex}_{tag}_a'] = np.nan; continue
                F[f'{nm}{ex}_{tag}_n'] = v[sel].mean(); F[f'{nm}{ex}_{tag}_a'] = np.average(v[sel], weights=a[sel])
            F[f'{nm}{ex}_mix'] = (np.average(v[ph], weights=a[ph]) * a[ph].sum() if ph.sum() else 0) / A
    F['inv_sqrt_N'] = 1 / np.sqrt(m.sum()); F['N_eff'] = (m & ~bd).sum() + 0.5 * (m & bd).sum(); F['bd_frac'] = bd[m].mean()
    F['dark_area_w'] = np.average(np.clip(dark[m], 0, 1), weights=a[m]); F['c4_a'] = np.average(c4[m], weights=a[m]); F['logar_a'] = np.average(np.log(ar[m]), weights=a[m])
    return F
FS = [feats(i) for i in ids]; keys = list(FS[0].keys()); X = np.array([[F[k] for k in keys] for F in FS], float)
rng = np.random.RandomState(2026); half = rng.rand(len(ids)) < 0.5
def pc(v, msk):
    ok = msk & np.isfinite(v); vr, rr, gg = stats.rankdata(v[ok]), stats.rankdata(res[ok]), stats.rankdata(pred[ok]); Dm = np.c_[np.ones(ok.sum()), gg]
    e1 = vr - Dm @ np.linalg.lstsq(Dm, vr, rcond=None)[0]; e2 = rr - Dm @ np.linalg.lstsq(Dm, rr, rcond=None)[0]; return np.corrcoef(e1, e2)[0, 1]
rows = sorted(((pc(X[:, j], clean & half), pc(X[:, j], clean & ~half), k) for j, k in enumerate(keys)), key=lambda t: -abs(t[0] + t[1]))
print(f'clean dev rows {clean.sum()}, residual RMSE {np.sqrt((res[clean] ** 2).mean()):.3f} (per-partition mean residual)')
print('top partial Spearman (half A, half B):'); [print(f'  {k:16s} {a:+.3f} {b:+.3f}') for a, b, k in rows[:12]]
N = np.array([F['N_eff'] for F in FS]); e2 = ((y - P) ** 2).mean(0)
Z = np.c_[np.ones(clean.sum()), 1 / N[clean]]; co = np.linalg.lstsq(Z, e2[clean], rcond=None)[0]
print(f'clean err^2 = {co[0]:.1f} + {co[1]:.0f}/N_eff   (median N_eff {np.median(N[clean]):.0f}; 1/N share of clean MSE {np.mean(co[1] / N[clean]) / e2[clean].mean():.2f})')
np.savez(_repo_path(H, 'exact-feats-dev.npz'), ids=np.array(ids), X=X, keys=np.array(keys))
