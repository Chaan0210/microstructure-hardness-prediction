import os, numpy as np, nsgain as N
from blend_dev import align_dev
y, g, nz = N.y, N.g, N.nz; base = align_dev('../push-800/dev-u1dmavgH2Z-anghcor.npz')
r = lambda a, m: float(np.sqrt(((a - y)[:, m] ** 2).mean()))
A = np.ones(400, bool); C = nz < 9.5
print(f'v3.5 (prod): all {r(base, A):.3f} clean {r(base, C):.3f} noisy {r(base, g):.3f}')
P = {'v3.5': base}
for m in ('v3', 'v2.6', 'v2.5-real'):
    f = f'dev-u1dmavgH2Z-anghcor-{m}.npz'
    if not os.path.exists(f): continue
    p = align_dev(f); P[m] = p; c = np.corrcoef((p - y).ravel(), (base - y).ravel())[0, 1]
    av = (base + p) / 2
    print(f'{m:9s}: all {r(p, A):.3f} clean {r(p, C):.3f} noisy {r(p, g):.3f} | err corr with v3.5 {c:.3f} | mean(v3.5,{m}) all {r(av, A):.3f} clean {r(av, C):.3f} noisy {r(av, g):.3f}')
if len(P) > 2:
    av = np.mean(list(P.values()), 0); print(f'mean of all {len(P)}: all {r(av, A):.3f} clean {r(av, C):.3f} noisy {r(av, g):.3f}')
