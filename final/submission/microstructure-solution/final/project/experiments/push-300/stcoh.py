import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi

def features(f0):
    row = []
    for s0 in (1.5, 3.0):
        f = ndi.gaussian_filter(f0, s0); gy, gx = np.gradient(f)
        for s1 in (8, 32):
            jxx = ndi.gaussian_filter(gx * gx, s1); jyy = ndi.gaussian_filter(gy * gy, s1); jxy = ndi.gaussian_filter(gx * gy, s1)
            cl = np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-9)
            G = np.hypot(jxx.mean() - jyy.mean(), 2 * jxy.mean()) / (jxx.mean() + jyy.mean())
            row += [G, float(np.median(cl))]
    return row

KEYS = [f'stcoh_{a}_{b}_{c}' for a in ('s1.5', 's3') for b in ('w8', 'w32') for c in ('glob', 'locmed')]

if __name__ == '__main__':
    split = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else f'stcoh-{split}.npz'
    P = Path(__file__).resolve().parents[2] / 'data'
    ids = [r['ID'] for r in csv.DictReader(open(P / ('train.csv' if split == 'train' else 'sample_submission.csv')))]
    X = np.array([features(np.asarray(Image.open(P / split / f'{i}.png'), np.float64)) for i in ids])
    np.savez(out, ids=np.array(ids), X=X, keys=np.array(KEYS)); print(out, X.shape)
