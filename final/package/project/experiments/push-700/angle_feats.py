import sys, numpy as np
split = sys.argv[1]
src = [f'../push-400/unet2maps-{n}-{split}.npz' for n in ('m1', 'm2', 'h1')] + [f'../push-400/unet4maps-z1-{split}.npz']
acc = None; ids = None
for f in src:
    M = np.load(f, allow_pickle=True); ii = [str(i) for i in M['ids']]
    if ids is None: ids = ii
    assert ii == ids
    maps = M['maps']; out = np.zeros((len(ids), 4))
    for n in range(len(ids)):
        m = maps[n].astype(np.float32); g = m[0] + m[1]; c2, s2 = m[5], m[6]
        C = (g * c2).sum() / g.sum(); S = (g * s2).sum() / g.sum(); vv = np.hypot(C, S) + 1e-9
        vp = np.hypot(c2, s2) + 1e-9
        out[n] = [(C * C - S * S) / vv, 2 * C * S / vv, (g * (c2 * c2 - s2 * s2) / vp).sum() / g.sum(), (g * 2 * c2 * s2 / vp).sum() / g.sum()]
    acc = out if acc is None else acc + out
    print(f, out[:3].round(3), flush=True)
acc /= len(src)
np.savez(f'angle-{split}.npz', ids=np.array(ids), X=acc, keys=np.array(['ang_c4', 'ang_s4', 'ang_c4_loc', 'ang_s4_loc']))
print('done', split, acc.mean(0).round(4), acc.std(0).round(4))
