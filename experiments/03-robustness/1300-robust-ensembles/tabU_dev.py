"""TabPFN slot (feature file pattern, 4 U-Net variants or one) with REALISTIC noise-uplift copies (degU1/degU2: real noisy train images + white
noise, label = source label) added to the context; copies whose source is in the validation fold are excluded. Official dev partitions/folds,
seeds 0,1,2 (100*s + fold as dev900). usage: tabU_dev.py VARIANT SLOTFILE_PATTERN SETS TAG   (SETS comma, or 'none' for the plain control)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300')); import harness
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200')); v, pat, sets, tag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
F = np.load(_repo_io(pat.format(v=v)), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float)
A98 = [str(k) for k in np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True)['keys']]
SX, SID = [], []
for s in ([] if sets == 'none' else sets.split(',')):
    S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); sk = [str(k) for k in S['keys']]
    cols = [sk.index(k) for k in keys]   # slot columns are a subset of the 98 anghcor columns
    SX.append(S['X'].astype(float)[:, cols]); SID += [str(i) for i in S['ids']]
SX = np.vstack(SX) if SX else np.zeros((0, XA.shape[1])); SID = np.array(SID)
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
save = {}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; g = np.zeros(len(y)); idsa = np.array(ids)
    for k in range(5):
        tr, va = fold != k, fold == k; vs = set(idsa[va]); kk = np.array([s not in vs for s in SID], bool)
        Xt = np.vstack([X[tr], SX[kk]]); yt = np.r_[y[tr], [lab[s] for s in SID[kk]]]
        g[va] = np.mean([harness.tabpfn_new('v3.5', 100 * sd + k, 'mps').fit(Xt, yt).predict(X[va]) for sd in (0, 1, 2)], 0)
    nzv = np.nan_to_num(X[:, keys.index('noise')] * 255, nan=0) if 'noise' in keys else np.zeros(len(y))
    print(name, 'grain %.4f noisy %.3f clean %.3f (copies %d)' % (np.sqrt(((g - y) ** 2).mean()), np.sqrt(((g - y)[nzv >= 12] ** 2).mean()), np.sqrt(((g - y)[nzv < 9.5] ** 2).mean()), len(SID)), flush=True)
    save.update({f'{name}_ids': idsa, f'{name}_y': y, f'{name}_fold': fold, f'{name}_g': g})
np.savez(_repo_io(f'dev-{v}-{tag}.npz'), **save)
