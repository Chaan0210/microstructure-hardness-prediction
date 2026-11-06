"""P2: realized vs parameter-level dependence. If hardness uses the generator parameter (e.g. phase probability) and the image shows a
finite sample of N grains, the slope of y on the realized measurement shrinks for small N (regression dilution ~ var_p/(var_p+c/N)).
Clean images only (precise measurement). Compare slopes across n_obj bins with bootstrap SE."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np
P = str(_REPO_ROOT) + '/'
F = np.load(_repo_io(P + 'experiments/push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float); ids = [str(i) for i in F['ids']]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io(P + 'data/train.csv')))}; y = np.array([lab[i] for i in ids])
c = lambda k: X[:, keys.index(k)]
nz = c('noise') * 255; N = c('n_obj')
ok = np.isfinite(nz) & np.isfinite(N) & (N >= 20)
clean = ok & (nz < np.nanmedian(nz))
regs = ['u_ph', 'het4_u', 'm_la_mean', 'm_el_pix', 'ang_c4', 'z_f25', 'z_f50']
for k in regs: print(k, 'nan', int(np.isnan(c(k)[clean]).sum()))
Z = np.c_[[c(k) for k in regs]].T
def fit(m):
    A = np.c_[np.ones(m.sum()), (Z[m] - Z[clean].mean(0)) / Z[clean].std(0)]
    return np.linalg.lstsq(A, y[m], rcond=None)[0][1:]
for tag, base in (('clean', clean), ('all', ok)):
    q = np.quantile(N[base], [1 / 3, 2 / 3]); b = np.digitize(N, q)
    print(f'--- {tag}: n_obj tercile cutpoints {q.round(0)}; standardized OLS coefs (per 1 SD of clean data), bootstrap SE')
    rng = np.random.RandomState(0)
    for j in range(3):
        m = base & (b == j); idx = np.where(m)[0]
        co = fit(m); bs = []
        for _ in range(300):
            s = rng.choice(idx, len(idx)); mm = np.zeros(len(y), bool); w = np.bincount(s, minlength=len(y))
            A = np.c_[np.ones(len(s)), (Z[s] - Z[clean].mean(0)) / Z[clean].std(0)]; bs.append(np.linalg.lstsq(A, y[s], rcond=None)[0][1:])
        se = np.std(bs, 0)
        print(f'bin {j} n={m.sum():3d} medN={np.median(N[m]):5.0f} ' + ' '.join(f'{k}:{v:+.2f}({s:.2f})' for k, v, s in zip(regs, co, se)))
# residual variance vs N in clean images under this linear model (+N terms)
m = clean; A = np.c_[np.ones(m.sum()), Z[m], np.log(N[m])]; r = y[m] - A @ np.linalg.lstsq(A, y[m], rcond=None)[0]
q = np.quantile(N[m], [0.2, 0.4, 0.6, 0.8]); b = np.digitize(N[m], q)
print('clean linear-model resid var by N quintile:', [(int(np.median(N[m][b == j])), round(float(np.mean(r[b == j] ** 2)), 1)) for j in range(5)])
