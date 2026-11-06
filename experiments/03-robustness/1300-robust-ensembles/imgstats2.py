from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, glob
exec(open(_repo_io('imgstats.py')).read().split("names = [")[0])
for s in ['deg', 'deg2', 'deg3', 'deg4', 'deg5', 'deg6', 'deg7', 'deg8', 'deg9', 'deg10']:
    fs = sorted(glob.glob(f'{P}/experiments/push-400/synth/train_{s}/*.png'))[:80]; S = np.array([stats(f) for f in fs])
    hf = S[:, 0]; keep = hf > 16.2   # comparable noise power to real noisy rows
    print(f'{s:6s} n {keep.sum():3d}  mid/hf {S[keep, 1].mean():.3f}  low/hf {S[keep, 2].mean():.3f}  grad p95 {S[keep, 6].mean():.2f}  bm p50-p10 {S[keep, 3].mean():.2f}')
R = np.array([stats(f'{P}/data/train/{ids[i]}.png') for i in np.where(nz >= 12)[0]]); keep = R[:, 0] > 16.2
print(f'real   n {keep.sum():3d}  mid/hf {R[keep, 1].mean():.3f}  low/hf {R[keep, 2].mean():.3f}  grad p95 {R[keep, 6].mean():.2f}  bm p50-p10 {R[keep, 3].mean():.2f}')
