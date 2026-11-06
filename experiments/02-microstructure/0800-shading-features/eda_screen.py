"""Split-half screen of battery features against the 044 dev residual (partial on the prediction g). Screen in half A, confirm in half B.
usage: eda_screen.py BATTERY_NPZ"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy import stats
R = np.load(_repo_io(__import__('os').environ.get('RESID', 'eda/resid044.npz')), allow_pickle=True); ids = [str(i) for i in R['ids']]; r = R['r']; g = R['g']; A = R['halfA']; X0 = R['X']; k0 = list(R['keys'])
B = np.load(_repo_io(sys.argv[1]), allow_pickle=True); bk = [str(k) for k in B['keys']]; bpos = {str(i): k for k, i in enumerate(B['ids'])}
Xb = B['X'][[bpos[i] for i in ids]].astype(float)
nz = X0[:, k0.index('noise')] * 255; clean = nz < 9.3
def pc(v, m):
    ok = m & np.isfinite(v)
    if ok.sum() < 30: return np.nan, ok.sum()
    vr = stats.rankdata(v[ok]); rr = stats.rankdata(r[ok]); gg = stats.rankdata(g[ok])
    D = np.c_[np.ones(ok.sum()), gg]; e1 = vr - D @ np.linalg.lstsq(D, vr, rcond=None)[0]; e2 = rr - D @ np.linalg.lstsq(D, rr, rcond=None)[0]
    return np.corrcoef(e1, e2)[0, 1], ok.sum()
print(f'{"feature":22s} | ALL: A      B      | CLEAN: A     B      | NOISY: A     B')
rows = []
for j, k in enumerate(bk):
    v = Xb[:, j]; out = []
    for m in (np.ones(len(r), bool), clean, ~clean):
        a, _ = pc(v, m & A); b_, _ = pc(v, m & ~A); out += [a, b_]
    rows.append((k, out))
    flag = any((abs(out[2 * s]) > 0.15) and (np.sign(out[2 * s]) == np.sign(out[2 * s + 1])) and abs(out[2 * s + 1]) > 0.1 for s in range(3))
    print(f'{k:22s} | {out[0]:+.3f} {out[1]:+.3f} | {out[2]:+.3f} {out[3]:+.3f} | {out[4]:+.3f} {out[5]:+.3f}' + ('   <==' if flag else ''))
print('SE of corr: all-half ~%.3f, clean-half ~%.3f' % (1 / np.sqrt(200), 1 / np.sqrt(clean.sum() / 2)))
