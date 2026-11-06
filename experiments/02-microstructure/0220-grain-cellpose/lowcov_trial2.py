"""Label-free re-trial with copied uint8 inputs: loose (cp -2) vs cp -3 / cp -4 on low- and high-coverage images."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, numpy as np
from pathlib import Path
from PIL import Image
from cellpose import models, utils
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
F = np.load(_repo_path(R, 'features-train.npz')); keys = list(F['keys']); cov = np.nan_to_num(F['X'][:, keys.index('coverage')], nan=0)
o = np.argsort(cov); ids = [F['ids'][i] for i in list(o[2:30:7]) + list(o[200:400:100])]
model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(R, 'weights/cpsam')))
cfg = {'cp-2': dict(flow_threshold=0.8, cellprob_threshold=-2), 'cp-3': dict(flow_threshold=1.0, cellprob_threshold=-3), 'cp-4': dict(flow_threshold=1.0, cellprob_threshold=-4)}
rows = []; out = {}
for i in ids:
    im = np.array(Image.open(_repo_path(P, f'data/train/{i}.png')))
    row = [np.stack([im] * 3, -1)]; out[i] = {}
    for k, c in cfg.items():
        m, _, _ = model.eval(im.copy(), **c)
        out[i][k] = (round(float((m > 0).mean()), 3), int(m.max()))
        rgb = np.stack([im] * 3, -1).copy()
        for ol in utils.outlines_list(m): rgb[ol[:, 1], ol[:, 0]] = [255, 40, 40]
        row.append(rgb)
    rows.append(np.concatenate(row, 1)); print(i, out[i], flush=True)
Image.fromarray(np.concatenate(rows[:3], 0)).save(_repo_path(R, 'lowcov-trial2-low.png')); Image.fromarray(np.concatenate(rows[3:], 0)).save(_repo_path(R, 'lowcov-trial2-mixed.png'))
json.dump(out, open(_repo_path(R, 'lowcov-trial2.json'), 'w'), indent=1)
