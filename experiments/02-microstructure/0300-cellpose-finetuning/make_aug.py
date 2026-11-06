"""Degradation-augmented train rows: for each source train image (noise below the train median), K degraded copies
with random (blur, noise, contrast); segment with the 220 v3 rule (+ optional preprocessing), extract v3 features.
Labels are unchanged (image noise is independent of hardness: Spearman -0.04). Test images are never used."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, argparse, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, _repo_io('../grain-cellpose-220'))
from pathlib import Path
from PIL import Image
from degrade import degrade
from seglib import seg
from features_v3 import extract_v3
ap = argparse.ArgumentParser(); ap.add_argument('--k', type=int, default=2); ap.add_argument('--out', default='aug-v3.npz'); ap.add_argument('--seed', type=int, default=11)
ap.add_argument('--views', type=int, default=1)
a = ap.parse_args()
P = Path(_repo_io('../..'))
F = np.load(_repo_io('../grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
nz = np.load(_repo_io('noise_train.npy')); src = np.where(nz <= np.median(nz))[0]
rng = np.random.RandomState(a.seed)
VIEWS = [lambda x: x, lambda x: np.ascontiguousarray(x.T), lambda x: np.ascontiguousarray(x[::-1]), lambda x: np.ascontiguousarray(np.rot90(x))][:a.views]
rows, src_idx, params = [], [], []
t0 = time.time()
for n, k in enumerate(src):
    im = np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png')))
    for c in range(a.k):
        sb, sn, ct = rng.uniform(0.6, 2.0), rng.uniform(10, 21), rng.uniform(0.7, 1.0)
        d = degrade(im, rng, blur=sb, noise=sn, contrast=ct)
        vals = []
        for fn in VIEWS:
            v = fn(d); f = extract_v3(v, seg(v)); vals.append(np.array([f.get(q, np.nan) if f else np.nan for q in keys]))
        rows.append(np.nanmean(np.vstack(vals), 0)); src_idx.append(k); params.append((sb, sn, ct))
    if (n + 1) % 10 == 0: print(json.dumps(dict(done=n + 1, n=len(src), sec=round(time.time() - t0))), flush=True)
np.savez(_repo_io(a.out), X=np.array(rows), src=np.array(src_idx), params=np.array(params), keys=np.array(keys))
print('DONE', flush=True)
