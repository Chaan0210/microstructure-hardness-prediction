"""Ablation of 061 on the Study900 dev (train-only). Level 1: components (leave-one-out with proportional renormalisation; and weight x1.5).
Level 2: TabPFN block members (3 slots x 4 U-Net variants; leave-one-slot/variant-out, single slot/variant). Per band dMSE (SE), per partition,
image bootstrap P(better). Noisy rows also slope-free (stretch removed)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import io_path as _repo_io
import numpy as np, sys
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import align_dev
import nsgain as N
y, nz, fail, c61, C, W = N.y, N.nz, N.fail, N.c61, N.C, N.W
f = fail[0]; bands = {'clean<11': (nz < 11) & ~f, '11-12': (nz >= 11) & (nz < 12) & ~f, 'noisy>=12': (nz >= 12) & ~f}
rng = np.random.RandomState(0); BI = [rng.randint(0, 400, 400) for _ in range(2000)]
def slopefree(new, m):
    o = new.copy()
    for k in range(3):
        x = c61[k, m] - c61[k, m].mean(); d = (new - c61)[k, m]; o[k, m] = c61[k, m] + d - (x @ d) / (x @ x) * x
    return o
def rep(nm, new):
    d = ((new - y) ** 2 - (c61 - y) ** 2); dm = d.mean(0); pb = np.mean([dm[b].mean() < 0 for b in BI])
    pp = np.sqrt(((c61 - y) ** 2).mean(1)) - np.sqrt(((new - y) ** 2).mean(1))
    cells = ' '.join(f'{k} {d[:, m].mean():+6.2f}({d[:, m].mean(0).std() / np.sqrt(m.sum()):.2f})' for k, m in bands.items())
    ns = slopefree(new, bands['noisy>=12']); dns = ((ns - y) ** 2 - (c61 - y) ** 2)[:, bands['noisy>=12']].mean()
    print(f'{nm:30s} gain {N.rm(c61) - N.rm(new):+.4f} ({"/".join(f"{v:+.3f}" for v in pp)}) P {pb:.2f} | {cells} | noisy slope-free {dns:+.2f}')
def blend(Cx, Wx): s = sum(Wx.values()); return np.where(fail, C['cnn'], sum(Wx[k] * Cx[k] for k in Wx) / s)
print('=== Level 1: components (061 dev %.4f)' % N.rm(c61))
for k in W:
    Wx = {j: w for j, w in W.items() if j != k}; rep(f'drop {k}', blend(C, Wx))
for k in W:
    Wx = dict(W); Wx[k] = W[k] * 1.5; rep(f'{k} x1.5', blend(C, Wx))
print('=== Level 2: TabPFN block members')
V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']; Sp = {'038': '../push-300/dev-{v}H2Z-v35.npz', '039': '../push-600/dev-{v}H2Z-angh.npz', '044': '../push-800/dev-{v}H2Z-anghcor.npz'}
G = {(s, v): align_dev(p.format(v=v)) for s, p in Sp.items() for v in V}
blk = np.mean(list(G.values()), 0); print('block reproduction max |diff| on non-failed rows', np.abs(blk - C['tab'])[:, ~f].max().round(6))
def withtab(T): Cx = dict(C); Cx['tab'] = T; return blend(Cx, W)
for s in Sp: rep(f'drop slot {s}', withtab(np.mean([G[(t, v)] for t in Sp for v in V if t != s], 0)))
for v in V: rep(f'drop variant {v}', withtab(np.mean([G[(t, u)] for t in Sp for u in V if u != v], 0)))
for s in Sp: rep(f'only slot {s}', withtab(np.mean([G[(s, v)] for v in V], 0)))
for v in V: rep(f'only variant {v}', withtab(np.mean([G[(t, v)] for t in Sp], 0)))
for s in Sp:
    for v in V: print(f'   member {s}/{v}: alone clean {np.sqrt(((G[(s, v)] - y) ** 2)[:, bands["clean<11"]].mean()):.3f} noisy {np.sqrt(((G[(s, v)] - y) ** 2)[:, bands["noisy>=12"]].mean()):.3f}')
