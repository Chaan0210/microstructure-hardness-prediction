from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, numpy as np
from pathlib import Path
from PIL import Image
from cellpose import models, utils
from skimage import restoration
R = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
ids = sys.argv[1].split(',')
model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(R, 'weights/cpsam')))
cfgs = {'default': dict(flow_threshold=0.4, cellprob_threshold=0.0, pre=None),
        'loose': dict(flow_threshold=0.8, cellprob_threshold=-2.0, pre=None),
        'tv+loose': dict(flow_threshold=0.8, cellprob_threshold=-2.0, pre='tv')}
tiles = []
for i in ids:
    im = np.array(Image.open(_repo_path(P, f'data/train/{i}.png')))
    row = [np.stack([im] * 3, -1)]
    for name, c in cfgs.items():
        x = im
        if c['pre'] == 'tv':
            f = im.astype(np.float32) / 255
            hh = (f[0::2, 0::2] - f[0::2, 1::2] - f[1::2, 0::2] + f[1::2, 1::2]) / 2; s = np.median(np.abs(hh)) / 0.6745
            x = (restoration.denoise_tv_chambolle(f, weight=max(0.02, 2.0 * s)) * 255).astype(np.float32)
        m, _, _ = model.eval(x, flow_threshold=c['flow_threshold'], cellprob_threshold=c['cellprob_threshold'])
        print(i, name, int(m.max()), round(float((m > 0).mean()), 3), flush=True)
        rgb = np.stack([im] * 3, -1).copy()
        for o in utils.outlines_list(m): rgb[o[:, 1], o[:, 0]] = [255, 40, 40]
        row.append(rgb)
    tiles.append(np.concatenate(row, 1))
Image.fromarray(np.concatenate(tiles, 0)).save(_repo_path(R, 'smoke-settings.png'))
