"""Label-free test-retest reliability of v3 features: segment transposed images (same rule as masks-v3) for 60 train images."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from PIL import Image
from cellpose import models
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]; sys.path.insert(0, str(R))
from features_v3 import extract_v3
z = np.load(_repo_path(R, 'masks-v3-train.npz')); ids = z['ids']; rng = np.random.default_rng(0); sel = rng.choice(len(ids), 60, replace=False)
model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(R, 'weights/cpsam')))
A, B = [], []
for k in sel:
    im = np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png')))
    m, _, _ = model.eval(np.ascontiguousarray(im.T).copy(), flow_threshold=0.8, cellprob_threshold=-2.0)
    if (m > 0).mean() < 0.95:
        m2, _, _ = model.eval(np.ascontiguousarray(im.T).copy(), flow_threshold=1.0, cellprob_threshold=-3.0)
        if (m2 > 0).mean() > (m > 0).mean(): m = m2
    fa = extract_v3(im, z['masks'][k]); fb = extract_v3(np.ascontiguousarray(im.T), m)
    if fa and fb: A.append(fa); B.append(fb)
keys = sorted(A[0]); res = {}
for key in keys:
    a = np.array([x.get(key, np.nan) for x in A]); b = np.array([x.get(key, np.nan) for x in B]); ok = np.isfinite(a) & np.isfinite(b)
    res[key] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 and a[ok].std() > 0 else None
json.dump(res, open(_repo_path(R, 'reliability.json'), 'w'), indent=1)
for k, v in sorted(res.items(), key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0)):
    print(f'{k:24s} {v}')
