"""Segmentation-free structure-tensor coherence features (8 columns) from the raw image, label-free.
pre-smoothing sigma s0 in (1.5, 3.0) x tensor window sigma s1 in (8, 32) -> global coherence and median of the local coherence map.
usage: stcoh.py SPLIT [OUT]   (SPLIT = train | test; ids in train.csv / sample_submission.csv order) -> stcoh-<split>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi

def features(f0):
    row = []
    for s0 in (1.5, 3.0):
        f = ndi.gaussian_filter(f0, s0); gy, gx = np.gradient(f)
        for s1 in (8, 32):
            jxx = ndi.gaussian_filter(gx * gx, s1); jyy = ndi.gaussian_filter(gy * gy, s1); jxy = ndi.gaussian_filter(gx * gy, s1)
            cl = np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-9)
            G = np.hypot(jxx.mean() - jyy.mean(), 2 * jxy.mean()) / (jxx.mean() + jyy.mean())
            row += [G, float(np.median(cl))]
    return row

KEYS = [f'stcoh_{a}_{b}_{c}' for a in ('s1.5', 's3') for b in ('w8', 'w32') for c in ('glob', 'locmed')]

if __name__ == '__main__':
    split = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else f'stcoh-{split}.npz'
    P = _repo_path(Path(__file__).resolve().parents[2], 'data')
    ids = [r['ID'] for r in csv.DictReader(open(_repo_io(P / ('train.csv' if split == 'train' else 'sample_submission.csv'))))]
    X = np.array([features(np.asarray(Image.open(_repo_io(P / split / f'{i}.png')), np.float64)) for i in ids])
    np.savez(_repo_io(out), ids=np.array(ids), X=X, keys=np.array(KEYS)); print(out, X.shape)
