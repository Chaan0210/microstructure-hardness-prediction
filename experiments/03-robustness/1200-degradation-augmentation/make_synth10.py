"""deg10: like deg4 (push-900/realdeg2 DEFAULT calibrated noisy preset, target noise from the train >= 12 pool) but the SOURCES are the train images
with train-median <= noise < 12 gray (new microstructures, labels known); their zero-shot masks-v3 drive the per-grain operations.
Label-free image transform of train data -> push-400/synth/train_deg10/<ID>.png"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, sys, numpy as np
from pathlib import Path
from PIL import Image
HERE = _repo_path(Path(__file__).resolve().parents[1], "push-400"); P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_path(HERE.parent, 'push-900'))); from realdeg2 import degrade2
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]
nz = np.load(_repo_path(HERE.parent, 'push-300/noise_train.npy')) * 255; med = np.median(nz)
M = np.load(_repo_path(HERE.parent, 'grain-cellpose-220/masks-v3-train.npz'))['masks']
clean = np.where((nz >= med) & (nz < 12.0))[0]; target_pool = np.sort(nz[nz >= 12.0])   # deg7: SOURCES are the mid-noise train images
PRM = {}   # realdeg2 DEFAULT = calibrated noisy preset (patchy boundary removal p~U(0.28, 0.68))
out = _repo_path(HERE, 'synth/train_deg10'); out.mkdir(parents=True, exist_ok=True); log = {}
for k in clean:
    r = np.random.RandomState(100_000 + k); a = np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png'))).astype(np.float64)
    tgt = float(target_pool[r.randint(len(target_pool))]); add = float(np.sqrt(max(tgt ** 2 - (0.4 * nz[k]) ** 2, 1.0)))
    g = degrade2(a, M[k], r, PRM, noise=add).astype(np.uint8); Image.fromarray(g).save(_repo_path(out, f'{ids[k]}.png'))
    hh = (g[0::2, 0::2].astype(float) - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    log[ids[k]] = dict(target_noise=tgt, added=add, measured_noise=float(np.median(np.abs(hh)) / 0.6745))
json.dump(dict(params=PRM, images=log), open(_repo_path(HERE, 'synth/train_deg10_meta.json'), 'w'), indent=0)
json.dump(log, open(_repo_path(HERE, 'synth/train_deg10_params.json'), 'w'), indent=0)
m = np.array([v['measured_noise'] for v in log.values()]); print(len(log), 'measured noise q', np.quantile(m, [.1, .5, .9]).round(2), ' real noisy half q', np.quantile(nz[nz >= med], [.1, .5, .9]).round(2))
