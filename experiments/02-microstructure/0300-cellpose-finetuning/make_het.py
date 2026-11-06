"""het_blocks features for a split: clean rows from zero-shot masks-v3, rows listed in an ft mask file from those masks (view 0).
usage: make_het.py SPLIT FT_MASKS_NPZ OUT"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from PIL import Image
from het_blocks import extract_het, KEYS
split, ftm, out = sys.argv[1], sys.argv[2], sys.argv[3]
M = np.load(_repo_io(f'../grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); ids = [str(i) for i in M['ids']]
Fm = np.load(_repo_io(ftm), allow_pickle=True); pos = {str(i): k for k, i in enumerate(Fm['ids'])}
Z = np.full((len(ids), len(KEYS)), np.nan); src = []
for n, i in enumerate(ids):
    im = np.array(Image.open(_repo_io(f'../../data/{split}/{i}.png')))
    m = Fm['masks'][pos[i]] if i in pos else M['masks'][n]; src.append('ft' if i in pos else 'zs')
    r = extract_het(im, m.astype(np.int32))
    if r: Z[n] = [r[k] for k in KEYS]
np.savez(_repo_io(out), ids=np.array(ids), X=Z, keys=np.array(KEYS), mask_source=np.array(src), ft_masks=ftm); print(out, Z.shape, int(np.isnan(Z).any(1).sum()))
