"""Probe: empirical-Bayes shrinkage of phase fraction toward the population mean by effective grain count (label-free)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X'])
A = np.load(_repo_io('angle-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
Neff = np.zeros(500)
for n in range(500):
    m = maps[n].astype(np.float32); g = m[0] + m[1]; a = np.exp(m[4])          # area (px, at full res) of the containing grain
    Neff[n] = 65536 / ((g * a).sum() / g.sum())                                 # 1/sum(w_i^2) for area weights
nz = X[:, keys.index('noise')] * 255; hq = nz < 11.9
for k in ['z_f25', 'u_ph', 'u_ph_hard']:
    p = X[:, keys.index(k)]; mu = p[hq].mean()
    samp = np.mean(p[hq] * (1 - p[hq]) / Neff[hq]); var_lat = max(p[hq].var() - samp, 1e-6)   # label-free variance decomposition (HQ rows)
    kk = np.mean(p * (1 - p)) / var_lat
    lam = Neff / (Neff + kk); ps = mu + lam * (p - mu)
    print(k, 'mu %.3f var_obs %.5f samp %.5f var_lat %.5f prior strength k=%.1f; Neff q %s' % (mu, p[hq].var(), samp, var_lat, kk, np.quantile(Neff, [.1, .5, .9]).round(0)))
    ctrl = ['het4_u', 'sizefield_std_s32', 'm_la_std', 'm_la_mean', 'm_el_pix', 'm_el_glob', 'n_obj', 'pore_frac']
    for nm, s in [('HQ', hq), ('HQ N<200', hq & (X[:, keys.index('n_obj')] < 200))]:
        C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(c) for c in ctrl]], A[s, :2]]
        def cv(v):
            Z = np.c_[C, v[s]]; yy = y[s]; e = []
            for r in range(10):
                f = np.arange(len(yy)) % 5; np.random.RandomState(r).shuffle(f)
                for q in range(5):
                    c = np.linalg.lstsq(Z[f != q], yy[f != q], rcond=None)[0]; e.append(np.mean((Z[f == q] @ c - yy[f == q]) ** 2))
            return np.sqrt(np.mean(e))
        print('   %-8s raw %.3f  shrunk %.3f' % (nm, cv(p), cv(ps)))
np.save(_repo_io('neff-train.npy'), Neff)
