import csv, json, sys, numpy as np
from pathlib import Path
from PIL import Image
HERE = Path(__file__).resolve().parents[1] / "push-400"; P = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'push-900')); from realdeg2 import degrade2
ids = [r['ID'] for r in csv.DictReader(open(P / 'data/train.csv'))]
nz = np.load(HERE.parent / 'push-300/noise_train.npy') * 255; med = np.median(nz)
M = np.load(HERE.parent / 'grain-cellpose-220/masks-v3-train.npz')['masks']
clean = np.where(nz < med)[0]; target_pool = np.sort(nz[nz >= 12.0])
PRM = {}   # realdeg2 DEFAULT = calibrated noisy preset (patchy boundary removal p~U(0.28, 0.68))
out = HERE / 'synth/train_deg4'; out.mkdir(parents=True, exist_ok=True); log = {}
for k in clean:
    r = np.random.RandomState(40_000 + k); a = np.array(Image.open(P / f'data/train/{ids[k]}.png')).astype(np.float64)
    tgt = float(target_pool[r.randint(len(target_pool))]); add = float(np.sqrt(max(tgt ** 2 - (0.4 * nz[k]) ** 2, 1.0)))
    g = degrade2(a, M[k], r, PRM, noise=add).astype(np.uint8); Image.fromarray(g).save(out / f'{ids[k]}.png')
    hh = (g[0::2, 0::2].astype(float) - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    log[ids[k]] = dict(target_noise=tgt, added=add, measured_noise=float(np.median(np.abs(hh)) / 0.6745))
json.dump(dict(params=PRM, images=log), open(HERE / 'synth/train_deg4_meta.json', 'w'), indent=0)
json.dump(log, open(HERE / 'synth/train_deg4_params.json', 'w'), indent=0)
m = np.array([v['measured_noise'] for v in log.values()]); print(len(log), 'measured noise q', np.quantile(m, [.1, .5, .9]).round(2), ' real noisy half q', np.quantile(nz[nz >= med], [.1, .5, .9]).round(2))
