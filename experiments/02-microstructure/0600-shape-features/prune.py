"""Label-free redundancy pruning: greedy in column order, drop a column if |Spearman| with an already-kept column >= THR (train features only).
usage: prune.py BASE_PREFIX THR OUT_PREFIX [extra columns to always drop, comma]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from scipy.stats import spearmanr
base, thr, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]; extra = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] else []
Tr = np.load(_repo_io(f'{base}-train.npz'), allow_pickle=True); keys = [str(k) for k in Tr['keys']]; X = Tr['X'].astype(float)
Xi = np.where(np.isnan(X), np.nanmedian(X, 0), X); C = np.abs(spearmanr(Xi).correlation); C = np.nan_to_num(C)
keep = []
for j in (range(len(keys) - 1, -1, -1) if __import__('os').environ.get('REV') else range(len(keys))):
    k = keys[j]
    if k in extra or k == 'n_obj' and False: continue
    if k != 'n_obj' and any(C[j, i] >= thr for i in keep): continue
    keep.append(j)
if keys.index('n_obj') not in keep: keep.append(keys.index('n_obj'))
keep = sorted(keep); dropped = [keys[j] for j in range(len(keys)) if j not in keep]
print(len(keys), '->', len(keep), 'dropped:', dropped)
for split in ('train', 'test'):
    F = np.load(_repo_io(f'{base}-{split}.npz'), allow_pickle=True)
    np.savez(_repo_io(f'{out}-{split}.npz'), X=F['X'][:, keep], keys=np.array([keys[j] for j in keep]), ids=F['ids'])
