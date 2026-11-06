"""degU1/degU2 (push-1300): noise-only uplift of the REAL noisy train images (measured noise >= 12): Gaussian white noise is added so the measured
noise moves to a target drawn from the train >= 12 pool above the source level (+1 gray). The real noisy rendering (blur, low contrast, missing
boundaries, per-grain scatter) is kept, so the copies match real images at the target noise on pixel statistics (push-1300/uplift_check.py).
Label-preserving transform of train data. usage: make_synthU.py 1|2 -> push-400/synth/train_degU<k>/<ID>.png"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, sys, numpy as np
from pathlib import Path
from PIL import Image
HERE = _repo_path(Path(__file__).resolve().parents[1], "push-400"); P = _RepoPath(__file__).resolve().parents[3]; K = int(sys.argv[1])
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]
nz = np.load(_repo_path(HERE.parent, 'push-300/noise_train.npy')) * 255; pool = np.sort(nz[nz >= 12.0])
out = _repo_path(HERE, f'synth/train_degU{K}'); out.mkdir(parents=True, exist_ok=True); log = {}
for k in np.where(nz >= 12.0)[0]:
    r = np.random.RandomState(90_000 + 1000 * K + k); cand = pool[pool > nz[k] + 1.0]
    if len(cand) == 0: continue
    tgt = float(cand[r.randint(len(cand))]); add = float(np.sqrt(tgt ** 2 - nz[k] ** 2))
    a = np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png'))).astype(np.float64); g = np.clip(np.round(a + r.randn(*a.shape) * add), 0, 255).astype(np.uint8)
    Image.fromarray(g).save(_repo_io(out / f'{ids[k]}.png'))
    hh = (g[0::2, 0::2].astype(float) - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    log[ids[k]] = dict(source_noise=float(nz[k]), target_noise=tgt, added=add, measured_noise=float(np.median(np.abs(hh)) / 0.6745))
json.dump(log, open(_repo_path(HERE, f'synth/train_degU{K}_params.json'), 'w'), indent=0)
m = np.array([v['measured_noise'] for v in log.values()]); t = np.array([v['target_noise'] for v in log.values()])
print(K, len(log), 'measured noise q', np.quantile(m, [.1, .5, .9]).round(2), 'target q', np.quantile(t, [.1, .5, .9]).round(2), 'mean |meas-target|', np.abs(m - t).mean().round(2))
