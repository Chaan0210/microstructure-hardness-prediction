"""Small physics-style linear model with generator-like features, CV on all / HQ rows (repeated 5-fold)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X']); ids = [str(i) for i in F['ids']]
Z = np.load(_repo_io('zones16-train.npy')); A = np.load(_repo_io('angle-train.npz'))['X']; E = np.load(_repo_io('extra-train.npz'))['X']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
g = lambda k: X[:, keys.index(k)]
nz = g('noise') * 255; lq = (nz >= 11.9).astype(float)
e = np.clip(g('m_el_pix'), 0, 0.97); ar = np.sqrt((1 + e) / (1 - e))
zn = Z[:, :, 2]
feats = {
 'ph': g('u_ph_hard'), 'zf25': g('z_f25'), 'zf50': g('z_f50'),
 'hp_zone': np.mean(np.exp(-0.5 * zn), 1), 'het_zone': zn.std(1), 'het4_u': g('het4_u'),
 'la': g('m_la_mean'), 'ar': ar, 'el_glob': g('m_el_glob'), 'ang_c4': A[:, 0], 'ang_s4': A[:, 1],
 'pore': g('px_pore_frac'), 'lq': lq, 'nz': nz,
}
def design(names, inter=()):
    cols = [feats[k] for k in names] + [feats[a] * feats[b] for a, b in inter]
    return np.column_stack(cols)
def cv(Xd, s, lam=1.0, reps=10):
    Xd = Xd[s]; yy = y[s]; e_ = []
    for r in range(reps):
        f = np.arange(len(yy)) % 5; np.random.RandomState(r).shuffle(f)
        for k in range(5):
            tr = f != k; mu, sd = Xd[tr].mean(0), Xd[tr].std(0) + 1e-9; A_ = (Xd[tr] - mu) / sd; B = (Xd[~tr] - mu) / sd
            c = np.linalg.solve(A_.T @ A_ + lam * np.eye(A_.shape[1]), A_.T @ (yy[tr] - yy[tr].mean())); e_.append(np.mean((B @ c + yy[tr].mean() - yy[~tr]) ** 2))
    return np.sqrt(np.mean(e_))
hq = nz < 11.9; al = nz > -1
base = ['ph', 'zf25', 'zf50', 'hp_zone', 'het_zone', 'het4_u', 'la', 'ar', 'el_glob', 'ang_c4', 'ang_s4', 'pore']
for nm, names, inter in [('base', base, ()), ('+ar*ang', base, (('ar', 'ang_c4'), ('ar', 'ang_s4'))), ('+ar*het', base, (('ar', 'het4_u'), ('ar', 'ang_c4'), ('ar', 'ang_s4'))),
                         ('+ar*la', base, (('ar', 'la'), ('ar', 'het4_u'), ('ar', 'ang_c4'), ('ar', 'ang_s4'))), ('+lq inter', base + ['lq', 'nz'], (('lq', 'ph'), ('lq', 'zf25'), ('lq', 'het4_u'), ('ar', 'ang_c4'), ('ar', 'ang_s4'), ('ar', 'la'), ('ar', 'het4_u')))]:
    print('%-10s HQ %.3f  all %.3f' % (nm, cv(design(names, inter), hq), cv(design(names, inter), al)))
