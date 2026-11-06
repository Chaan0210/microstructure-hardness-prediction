"""Masks v3: keep v2 ('loose', cp -2) masks; images with coverage < 0.95 are re-run with cp -3 / flow 1.0 and replaced if coverage increases.
Rule fixed from label-free visual trials (lowcov-trial2-*.png): identical output on well-segmented images, fuller tilings on noisy ones."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, time, numpy as np
from pathlib import Path
from PIL import Image
from cellpose import models
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
split = sys.argv[1]; out = _repo_path(R, f'masks-v3-{split}.npz'); assert not out.exists()
z = np.load(_repo_path(R, f'masks-{split}.npz')); ids, masks = z['ids'], z['masks'].copy()
model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(R, 'weights/cpsam')))
cov = (masks > 0).mean((1, 2)); todo = np.where(cov < 0.95)[0]; log = []; t0 = time.time()
for n, k in enumerate(todo):
    im = np.array(Image.open(_repo_path(P, f'data/{split}/{ids[k]}.png')))
    m, _, _ = model.eval(im.copy(), flow_threshold=1.0, cellprob_threshold=-3.0)
    c = float((m > 0).mean()); rep = c > cov[k]
    if rep: masks[k] = m
    log.append(dict(id=str(ids[k]), cov_v2=float(cov[k]), cov_cp3=c, replaced=bool(rep)))
    if (n + 1) % 20 == 0: print(json.dumps(dict(done=n + 1, n=len(todo), sec=round(time.time() - t0))), flush=True)
np.savez_compressed(_repo_io(out), ids=ids, masks=masks)
json.dump(dict(split=split, n_rerun=len(todo), n_replaced=int(sum(l['replaced'] for l in log)), items=log), open(_repo_path(R, f'segment-v3-{split}.json'), 'w'), indent=1)
print('DONE', len(todo), flush=True)
