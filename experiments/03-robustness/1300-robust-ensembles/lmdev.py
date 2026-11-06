"""Label-matched dev (LM-dev): dev noisy rows reweighted by w(y) = p_clean(y) / p_noisy(y) (Gaussian KDE on the 500 train labels: clean = noise < 9.5,
noisy = noise >= 12), so the noisy rows' label distribution matches the clean rows' (microstructure is independent of the imaging preset). Clean and mid
rows unweighted. Train labels only. Reports weighted dev RMSE change and weighted noisy dMSE (SE) for the candidates with Public readings."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, glob, numpy as np
from scipy.stats import gaussian_kde
src = open(_repo_io('kdiag.py')).read(); src = src[:src.index("u = (c57 - c53)")]; exec(compile(src, 'kdiag', 'exec'))
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; yall = np.array([lab[str(i)] for i in F['ids']])
nzall = np.nan_to_num(F['X'][:, keys.index('noise')].astype(float) * 255, nan=0)
kc, kn = gaussian_kde(yall[nzall < 9.5]), gaussian_kde(yall[nzall >= 12])
w = np.ones(400); w[m] = kc(y[m]) / kn(y[m]); w[m] *= m.sum() / w[m].sum()
print('noisy weights: ESS/n %.2f, range %.2f-%.2f; weighted noisy y sd %.1f (unweighted %.1f, clean %.1f)' % (w[m].sum() ** 2 / (w[m] ** 2).sum() / m.sum(), w[m].min(), w[m].max(),
      np.sqrt(np.average((y[m] - np.average(y[m], weights=w[m])) ** 2, weights=w[m])), y[m].std(), y[nz < 9.5].std()))
c67 = c61 + 0.128 * m * ts
A = np.load(_repo_io('../push-1300/audit061.npz')); cnn = A['C_cnn']; c68 = np.where(m, (c61 - 0.16 * cnn) / 0.84, c61)
wr = lambda a: float(np.sqrt((w * ((a - y) ** 2)).mean()))
def rep(nm, base, new, pub):
    dm = (w * ((new - y) ** 2 - (base - y) ** 2))[:, m].mean(0); se = dm.std() / np.sqrt(m.sum())
    print(f'{nm:9s} LM-dev {wr(base):.4f} -> {wr(new):.4f} ({wr(base) - wr(new):+.4f}; plain {np.sqrt(((base - y) ** 2).mean()) - np.sqrt(((new - y) ** 2).mean()):+.4f})  weighted noisy dMSE {dm.mean():+.2f} SE {se:.2f}   Public {pub}')
rep('053->057', c53, c57, 'better (-0.042)'); rep('053->061', c53, c61, 'better (-0.055)'); rep('061->062', c61, c62, 'n/a'); rep('061->064', c61, c64, 'worse (+0.034)')
rep('061->067', c61, c67, 'n/a'); rep('061->068', c61, c68, 'n/a'); rep('062->064', c62, c64, 'n/a')
np.save(_repo_io('../push-1300/lm_weights.npy'), w)
