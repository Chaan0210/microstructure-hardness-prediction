# deg12 (push-1300): deg4 with the degradation re-calibrated on label-free pixel statistics (boundary removal U(0.8,1.0), blur 0.9, dark contrast default 0.675)
"""deg12: degrade each clean TRAIN image (noise < train median) once with push-900/realdeg2 DEFAULT parameters (calibrated noisy preset fitted on
our train images only: patchy boundary-line removal p~U(0.28, 0.68), per-grain phase contrast, matrix spread, blur, shading, pore depth), target total noise
drawn from the train images with noise >= 12 gray. Label-free image transform of train data -> push-400/synth/train_deg12/<ID>.png"""
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
clean = np.where(nz < med)[0]; target_pool = np.sort(nz[nz >= 12.0])
PRM = dict(P_LO=0.8, P_HI=1.0, BLUR_MU=0.9)   # push-1300 ucalib.py: measurement-level (U-Net marginals) + pixel calibration best balance (U 0.52->0.20, pix 1.49->0.39)
out = _repo_path(HERE, 'synth/train_deg12'); out.mkdir(parents=True, exist_ok=True); log = {}
for k in clean:
    r = np.random.RandomState(120_000 + k); a = np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png'))).astype(np.float64)
    tgt = float(target_pool[r.randint(len(target_pool))]); add = float(np.sqrt(max(tgt ** 2 - (0.4 * nz[k]) ** 2, 1.0)))
    g = degrade2(a, M[k], r, PRM, noise=add).astype(np.uint8); Image.fromarray(g).save(_repo_path(out, f'{ids[k]}.png'))
    hh = (g[0::2, 0::2].astype(float) - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    log[ids[k]] = dict(target_noise=tgt, added=add, measured_noise=float(np.median(np.abs(hh)) / 0.6745))
json.dump(dict(params=PRM, images=log), open(_repo_path(HERE, 'synth/train_deg12_meta.json'), 'w'), indent=0)
json.dump(log, open(_repo_path(HERE, 'synth/train_deg12_params.json'), 'w'), indent=0)
m = np.array([v['measured_noise'] for v in log.values()]); print(len(log), 'measured noise q', np.quantile(m, [.1, .5, .9]).round(2), ' real noisy half q', np.quantile(nz[nz >= med], [.1, .5, .9]).round(2))
