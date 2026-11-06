"""Pixel statistics (imgstats.stats, label-free) for all train images and all degraded copy images (deg..deg10)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, glob
from pathlib import Path
exec(open(_repo_io('imgstats.py')).read().split("names = [")[0])
out = {'train_ids': np.array(ids), 'train_X': np.array([stats(f'{P}/data/train/{i}.png') for i in ids])}
for s in ['deg', 'deg2', 'deg3', 'deg4', 'deg5', 'deg6', 'deg7', 'deg8', 'deg9', 'deg10']:
    fs = sorted(glob.glob(f'{P}/experiments/push-400/synth/train_{s}/*.png')); out[f'{s}_ids'] = np.array([Path(f).stem for f in fs]); out[f'{s}_X'] = np.array([stats(f) for f in fs]); print(s, len(fs), flush=True)
np.savez(_repo_io('pixstats.npz'), **out)
