"""Compliant label-preserving degraded copies of clean TRAIN images (noise < train median) for the DTA partners.
Only generic whole-image transforms, no analysis result (no masks, no segmentation) touches the image:
  mask-free smooth random brightness field (local brightness jitter), quadratic shading (illumination), global dark-tone compression
  (tone curve: values below the image median are pulled toward it by factor c_d, a contrast adjustment), extra Gaussian blur, global contrast/brightness,
  optional patchy grey closing (standard 3x3 morphological filter blended through a mask-free random-field mask: weakens thin dark lines),
  white noise topped up to a target total noise drawn from the train images with noise >= 12.
usage: make_copies_c.py SET SEED CLOSE(0|1)  -> push-400/synth/train_<SET>/<ID>.png + train_<SET>_params.json"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, sys, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
E = Path(__file__).resolve().parents[1]; P = E.parent; S = _repo_path(E, 'push-400/synth')
st, seed, close = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]
nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255; med = np.median(nz); clean = np.where(nz < med)[0]; pool = np.sort(nz[nz >= 12.0])
YY, XX = np.mgrid[0:256, 0:256] / 255.0 - 0.5
def rfield(r, lo, hi):
    g = ndi.gaussian_filter(r.normal(0, 1, (256, 256)), r.uniform(lo, hi)); return g / max(g.std(), 1e-6)
out = _repo_path(S, f'train_{st}'); out.mkdir(parents=True, exist_ok=True); log = {}
for k in clean:
    r = np.random.RandomState(seed * 1000 + k); a = np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png'))).astype(np.float64); f = a.copy(); prm = {}
    prm['jitter'] = r.uniform(2, 6); f = f + prm['jitter'] * rfield(r, 2.5, 8.0)
    m = np.median(f); prm['c_d'] = float(np.clip(r.normal(0.675, 0.15), 0.35, 1.0)); f = np.where(f < m, m - prm['c_d'] * (m - f), f)
    if close:
        prm['p_close'] = r.uniform(0.28, 0.68); thr = np.quantile(fm := rfield(r, 6.0, 16.0), 1 - prm['p_close'])
        w = ndi.gaussian_filter((fm > thr).astype(float), 1.5); f = (1 - w) * f + w * ndi.grey_closing(f, size=(3, 3))
    cq = r.normal(0, 1, 5); sh = cq[0] * YY + cq[1] * XX + cq[2] * YY * YY + cq[3] * XX * XX + cq[4] * XX * YY; prm['shade'] = r.uniform(0, 0.05)
    f = f * (1 + prm['shade'] * sh / max(np.abs(sh).max(), 1e-6))
    prm['blur'] = abs(r.normal(0.75, 0.15)); f = ndi.gaussian_filter(f, prm['blur'])
    prm['contrast'] = r.uniform(0.75, 1.0); mu = f.mean(); f = mu + prm['contrast'] * (f - mu) + r.uniform(-8, 8)
    tgt = float(pool[r.randint(len(pool))]); add = float(np.sqrt(max(tgt ** 2 - (0.4 * nz[k]) ** 2, 1.0))); f = f + r.normal(0, add, f.shape)
    g = np.clip(np.round(f), 0, 255).astype(np.uint8); Image.fromarray(g).save(_repo_io(out / f'{ids[k]}.png'))
    hh = (g[0::2, 0::2].astype(float) - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    log[ids[k]] = dict(prm, target_noise=tgt, added=add, measured_noise=float(np.median(np.abs(hh)) / 0.6745))
json.dump(log, open(_repo_path(S, f'train_{st}_params.json'), 'w'), indent=0)
mm = np.array([v['measured_noise'] for v in log.values()]); print(st, len(log), 'measured noise q', np.quantile(mm, [.1, .5, .9]).round(2), 'real >=12 q', np.quantile(pool, [.1, .5, .9]).round(2))
