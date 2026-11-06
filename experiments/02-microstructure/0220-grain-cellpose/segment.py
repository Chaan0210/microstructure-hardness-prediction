"""Segment all images of one split with Cellpose-SAM (fixed 'loose' config chosen on 3 smoke images before any label use)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, numpy as np
from pathlib import Path
from PIL import Image
from cellpose import models
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
split = sys.argv[1]  # 'train' (test only after a candidate passes)
assert split in ('train', 'test')
files = sorted((_repo_path(P, f'data/{split}')).glob('*.png'))
out = _repo_path(R, f'masks-{split}.npz')
assert not out.exists(), 'refuse overwrite'
CFG = dict(flow_threshold=0.8, cellprob_threshold=-2.0)
model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(R, 'weights/cpsam')))
masks = np.zeros((len(files), 256, 256), np.uint16); t0 = time.time()
for k, f in enumerate(files):
    im = np.array(Image.open(_repo_io(f)))
    m, _, _ = model.eval(im, **CFG)
    masks[k] = m
    if (k + 1) % 25 == 0: print(json.dumps(dict(done=k + 1, n=len(files), sec=round(time.time() - t0))), flush=True)
np.savez_compressed(_repo_io(out), ids=np.array([f.stem for f in files]), masks=masks)
json.dump(dict(split=split, n=len(files), config=CFG, seconds=time.time() - t0, weights_sha256='e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2'), open(_repo_path(R, f'segment-{split}.json'), 'w'), indent=1)
print('DONE', flush=True)
