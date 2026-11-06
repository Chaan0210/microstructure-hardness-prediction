"""zrobust features for train/test rows: clean rows from zero-shot masks-v3, noisy rows from a saved ft mask file (view 0)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from PIL import Image
from zrobust import extract_r, KEYS
split, ftmasks, out = sys.argv[1], sys.argv[2], sys.argv[3]
M = np.load(_repo_io(f'../grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); ids = [str(i) for i in M['ids']]
Fm = np.load(_repo_io(ftmasks), allow_pickle=True); ftm = {str(i): k for k, i in enumerate(Fm['ids'])}
Z = np.full((len(ids), len(KEYS)), np.nan)
for n, i in enumerate(ids):
    im = np.array(Image.open(_repo_io(f'../../data/{split}/{i}.png')))
    m = Fm['masks'][ftm[i]] if i in ftm else M['masks'][n]
    r = extract_r(im, m.astype(np.int32))
    if r: Z[n] = [r[k] for k in KEYS]
np.savez(_repo_io(out), ids=np.array(ids), Z=Z, keys=np.array(KEYS), ftmasks=ftmasks); print(out, Z.shape, np.isnan(Z).any(1).sum())
