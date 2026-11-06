"""Average v3 features over TTA views for noisy images -> features-v3tta-{split}.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from PIL import Image
from features_v3 import extract_v3
VIEWS = {'T': lambda a: np.ascontiguousarray(a.T), 'UD': lambda a: np.ascontiguousarray(a[::-1]), 'R90': lambda a: np.ascontiguousarray(np.rot90(a))}
for split in sys.argv[1:]:
    F = np.load(_repo_io(f'features-v3-{split}.npz')); keys = list(F['keys']); X = F['X'].copy(); pos = {i: k for k, i in enumerate(F['ids'])}
    T = np.load(_repo_io(f'masks-tta-{split}.npz'))
    for n, i in enumerate(T['ids']):
        im = np.array(Image.open(_repo_io(f'../../data/{split}/{i}.png'))); vals = [X[pos[i]]]
        for v, fn in VIEWS.items():
            f = extract_v3(fn(im), T[v][n]); vals.append(np.array([f.get(k, np.nan) if f else np.nan for k in keys]))
        X[pos[i]] = np.nanmean(np.vstack(vals), 0)
    np.savez(_repo_io(f'features-v3tta-{split}.npz'), ids=F['ids'], X=X, keys=np.array(keys), tta_ids=T['ids']); print(split, 'tta images', len(T['ids']), flush=True)
