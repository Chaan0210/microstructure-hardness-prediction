from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from cellpose import models, utils
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
ids = sys.argv[1].split(',')
model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(R, 'weights/cpsam')))
print('device', model.device, flush=True)
tiles = []; out = {}
for i in ids:
    im = np.array(Image.open(_repo_path(P, f'data/train/{i}.png')))
    t = time.time(); m, flows, styles = model.eval(im, flow_threshold=0.4, cellprob_threshold=0.0)
    dt = time.time() - t
    n = int(m.max()); cov = float((m > 0).mean())
    out[i] = dict(n=n, coverage=cov, sec=round(dt, 2))
    print(i, out[i], flush=True)
    rgb = np.stack([im] * 3, -1).copy()
    for o in utils.outlines_list(m):
        rgb[o[:, 1], o[:, 0]] = [255, 40, 40]
    tiles.append(np.concatenate([np.stack([im] * 3, -1), rgb], 1))
rows = [np.concatenate(tiles[k:k + 2], 1) for k in range(0, len(tiles), 2)]
Image.fromarray(np.concatenate(rows, 0)).save(_repo_path(R, 'smoke-overlay.png'))
json.dump(out, open(_repo_path(R, 'smoke.json'), 'w'), indent=1)
