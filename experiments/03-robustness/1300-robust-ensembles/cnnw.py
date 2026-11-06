"""CNN021 share in the 061 blend (train-only dev): scale CNN weight w (others renormalised proportionally), per band, per partition, slope-free (stretch removed)."""
import numpy as np
from nsgain import y, nz, fail, C, W, c61, rm
f = fail[0]; bands = {'clean<11': (nz < 11) & ~f, '11-12': (nz >= 11) & (nz < 12) & ~f, 'noisy>=12': (nz >= 12) & ~f}
rest = sum(W[k] * C[k] for k in W if k != 'cnn') / (1 - W['cnn'])
def blend(w, mask=None):
    b = w * C['cnn'] + (1 - w) * rest
    if mask is not None: b = np.where(mask, b, c61)
    return np.where(fail, C['cnn'], b)
assert np.abs(blend(0.16) - c61).max() < 1e-9
def slopefree(new, mask):
    o = new.copy()
    for k in range(3):
        x = c61[k, mask] - c61[k, mask].mean(); d = (new - c61)[k, mask]; o[k, mask] = c61[k, mask] + d - (x @ d) / (x @ x) * x
    return o
print('CNN alone', round(rm(C['cnn']), 3), ' rest alone', round(rm(np.where(fail, C['cnn'], rest)), 3))
for nm, m in list(bands.items()) + [('all', ~f)]:
    print(f'--- band {nm} (n={m.sum()})')
    for w in (0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24):
        b = blend(w, m); bs = slopefree(b, m)
        pp = np.sqrt(((c61 - y) ** 2).mean(1)) - np.sqrt(((b - y) ** 2).mean(1)); ps = np.sqrt(((c61 - y) ** 2).mean(1)) - np.sqrt(((bs - y) ** 2).mean(1))
        dm = ((b - y) ** 2 - (c61 - y) ** 2)[:, m].mean(0); se = dm.std() / np.sqrt(m.sum())
        print(f'w {w:.2f}: gain {rm(c61) - rm(b):+.4f} parts {"/".join(f"{v:+.3f}" for v in pp)} | slope-free {rm(c61) - rm(bs):+.4f} parts {"/".join(f"{v:+.3f}" for v in ps)} | band dMSE {dm.mean():+.2f} SE {se:.2f}')
