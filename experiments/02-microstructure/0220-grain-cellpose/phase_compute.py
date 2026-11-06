from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from PIL import Image
from phase_features import grain_table, phase_estimates
KEEP = ['ph_count_z12', 'ph_count_z18', 'ph_count_z25', 'ph_area_z12', 'ph_area_z18', 'ph_area_z25', 'ph_mirror_count_15', 'ph_mirror_area_15', 'ph_gmm_count', 'ph_gmm_area']
for split in ['train', 'test']:
    z = np.load(_repo_io(f'masks-v3-{split}.npz')); X = []
    for i, m in zip(z['ids'], z['masks']):
        t = grain_table(np.array(Image.open(_repo_io(f'../../data/{split}/{i}.png'))), m)
        e = phase_estimates(t) if t is not None else None
        X.append([e[k] if e else np.nan for k in KEEP])
    np.savez(_repo_io(f'phase-{split}.npz'), ids=z['ids'], X=np.array(X, float), keys=np.array(KEEP)); print(split, np.array(X).shape, flush=True)
