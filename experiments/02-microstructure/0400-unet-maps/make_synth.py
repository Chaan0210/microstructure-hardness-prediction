"""Degrade each clean TRAIN image (noise < train median) once to mimic the real noisy half (label-free image transform of train data).
Per image (seeded by index): grain intensity scatter (zero-shot masks), smooth shading, extra blur, contrast, brightness, white noise topped up to a
target total noise drawn from the empirical noisy-half noise distribution. Writes PNGs to synth/train_deg/<ID>.png."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
HERE = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]
nz = np.load(_repo_path(HERE.parent, 'push-300/noise_train.npy')) * 255; med = np.median(nz)
M = np.load(_repo_path(HERE.parent, 'grain-cellpose-220/masks-v3-train.npz'))['masks']
clean = np.where(nz < med)[0]; target_pool = np.sort(nz[nz >= med])
out = _repo_path(HERE, 'synth/train_deg'); out.mkdir(parents=True, exist_ok=True)
yy, xx = np.mgrid[0:256, 0:256] / 255.0 - 0.5; log = {}
for k in clean:
    r = np.random.RandomState(10_000 + k); a = np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png'))).astype(np.float64); lab = M[k].astype(np.int64)
    off = r.normal(0, r.uniform(2, 6), lab.max() + 1); off[0] = 0; f = a + off[lab]
    cq = r.normal(0, 1, 5); sh = cq[0] * yy + cq[1] * xx + cq[2] * yy * yy + cq[3] * xx * xx + cq[4] * xx * yy; f = f + r.uniform(0, 6) * sh / max(sh.std(), 1e-6)
    sb = r.uniform(0.8, 1.6); f = ndi.gaussian_filter(f, sb)
    c = r.uniform(0.65, 0.95); mu = f.mean(); f = mu + c * (f - mu) + r.uniform(-10, 10)
    # remaining existing noise after blur ~ nz/(2*sqrt(pi)*sb) ; top up to target
    tgt = target_pool[r.randint(len(target_pool))]; exist = nz[k] * c / (2 * np.sqrt(np.pi) * sb); add = np.sqrt(max(tgt ** 2 - exist ** 2, 1.0))
    f = f + r.normal(0, add, f.shape); g = np.clip(np.round(f), 0, 255).astype(np.uint8)
    Image.fromarray(g).save(_repo_path(out, f'{ids[k]}.png'))
    hh = (g[0::2, 0::2].astype(float) - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    log[ids[k]] = dict(blur=sb, contrast=c, target_noise=float(tgt), added=float(add), measured_noise=float(np.median(np.abs(hh)) / 0.6745))
json.dump(log, open(_repo_path(HERE, 'synth/train_deg_params.json'), 'w'), indent=0)
m = np.array([v['measured_noise'] for v in log.values()]); print(len(log), 'measured noise of synthetic q', np.quantile(m, [.1, .5, .9]).round(2), ' real noisy half q', np.quantile(nz[nz >= med], [.1, .5, .9]).round(2))
