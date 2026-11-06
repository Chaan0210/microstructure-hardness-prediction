"""Domain-wise quantile alignment + degradation transfer for TabPFN (044 slot features, one U-Net variant). Per outer fold, every feature is mapped
to its normal-score within its domain (real clean < 9.5, real mid 9.5-12, real noisy >= 12, degraded copies), maps fitted on outer-training rows /
allowed copies only (microstructure distribution is the same across domains, so within-domain quantiles put the instruments on one scale); the raw
noise column is kept. TabPFN context = real outer-training rows [+ copies]. usage: tabQ_dev.py VARIANT SETS|none TAG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, warnings, numpy as np
from pathlib import Path
from scipy.stats import norm
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300')); import harness
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); v, sets, tag = sys.argv[1], sys.argv[2], sys.argv[3]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
SX, SID = [], []
for s in ([] if sets == 'none' else sets.split(',')):
    S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
SX = np.vstack(SX) if SX else np.zeros((0, XA.shape[1])); SID = np.array(SID)
def dom(X): nz = np.nan_to_num(X[:, ni] * 255, nan=0); return np.where(nz < 9.5, 0, np.where(nz < 12, 1, 2))
def fit_q(Xref):   # per-feature sorted reference values (NaN-aware)
    return [np.sort(Xref[np.isfinite(Xref[:, j]), j]) for j in range(Xref.shape[1])]
def apply_q(X, Q):
    Z = X.copy()
    for j, r in enumerate(Q):
        if j == ni or len(r) < 5: continue
        x = X[:, j]; ok = np.isfinite(x); u = (np.searchsorted(r, x[ok], 'left') + np.searchsorted(r, x[ok], 'right')) / 2 / len(r)
        Z[ok, j] = norm.ppf(np.clip(u, 0.5 / len(r), 1 - 0.5 / len(r)))
    return Z
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
save = {}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; D = dom(X); g = np.zeros(len(y)); idsa = np.array(ids)
    for k in range(5):
        tr, va = fold != k, fold == k; vs = set(idsa[va]); kk = np.array([s not in vs for s in SID], bool)
        Qs = {c: fit_q(X[tr & (D == c)]) for c in (0, 1, 2)}; Z = X.copy()
        for c in (0, 1, 2): Z[D == c] = apply_q(X[D == c], Qs[c])
        Zc = apply_q(SX[kk], fit_q(SX[kk])) if kk.sum() else np.zeros((0, X.shape[1]))
        Xt = np.vstack([Z[tr], Zc]); yt = np.r_[y[tr], [lab[s] for s in SID[kk]]]
        g[va] = np.mean([harness.tabpfn_new('v3.5', 100 * sd + k, 'mps').fit(Xt, yt).predict(Z[va]) for sd in (0, 1, 2)], 0)
    nzv = np.nan_to_num(X[:, ni] * 255, nan=0)
    print(name, 'grain %.4f noisy %.3f clean %.3f mid %.3f (copies %d)' % (np.sqrt(((g - y) ** 2).mean()), np.sqrt(((g - y)[nzv >= 12] ** 2).mean()), np.sqrt(((g - y)[nzv < 9.5] ** 2).mean()), np.sqrt(((g - y)[(nzv >= 9.5) & (nzv < 12)] ** 2).mean()), len(SID)), flush=True)
    save.update({f'{name}_ids': idsa, f'{name}_y': y, f'{name}_fold': fold, f'{name}_g': g})
np.savez(_repo_io(f'dev-{v}-{tag}.npz'), **save)
