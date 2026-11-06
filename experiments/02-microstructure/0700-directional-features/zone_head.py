"""Q5 (review proposal): small hardness heads trained directly on outer-fold labels (no residuals). Inputs: 97 angh columns (+ 4x4 zone array of 6 channels).
A: MLP on 97 cols. B: + real zone array through 2 small conv layers (position kept, flattened). C: + zone array with the 16 zones permuted per image
(all 6 channels move together: zone content kept, position/neighbourhood destroyed). Blend rule fixed in advance: grain = 0.8 * 039 + 0.2 * head.
usage: zone_head.py PARTITION"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, torch, torch.nn as nn
part = sys.argv[1]; torch.set_num_threads(6)
D = np.load(_repo_io('../push-600/dev-u1dmavgH2Z-angh.npz'), allow_pickle=True)
ids = [str(i) for i in D[part + '_ids']]; y = D[part + '_y'].astype(np.float32); fold = D[part + '_fold']; p021 = D[part + '_p021']; failed = D[part + '_failed']
F = np.load(_repo_io('feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); fp = {str(i): k for k, i in enumerate(F['ids'])}; X = np.nan_to_num(F['X'][[fp[i] for i in ids]].astype(np.float32))
ZV = np.load(_repo_io('zonevec-train.npz'), allow_pickle=True); zp = {str(i): k for k, i in enumerate(ZV['ids'])}; Z = ZV['Z'][[zp[i] for i in ids]].astype(np.float32)   # [N,16,6]
rng = np.random.RandomState(11); Zs = np.stack([Z[n, rng.permutation(16)] for n in range(len(ids))])
toimg = lambda A: np.transpose(A.reshape(-1, 4, 4, 6), (0, 3, 1, 2)).copy()
class Head(nn.Module):
    def __init__(s, nx, zone):
        super().__init__(); s.zone = zone
        if zone: s.conv = nn.Sequential(nn.Conv2d(6, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 4, 3, padding=1), nn.GELU())
        s.mlp = nn.Sequential(nn.Linear(nx + (64 if zone else 0), 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(s, x, z=None):
        h = x if not s.zone else torch.cat([x, s.conv(z).flatten(1)], 1); return s.mlp(h)[:, 0]
def run(arm, tr, va, seed):
    torch.manual_seed(seed); r = np.random.RandomState(seed)
    mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mx) / sx
    zz = None if arm == 'A' else (Z if arm == 'B' else Zs); zt = None
    if zz is not None:
        mz, sz = zz[tr].reshape(-1, 6).mean(0), zz[tr].reshape(-1, 6).std(0) + 1e-6; zt = torch.tensor(toimg((zz - mz) / sz))
    my, sy = y[tr].mean(), y[tr].std(); yt = torch.tensor((y - my) / sy); xt = torch.tensor(Xn)
    perm = r.permutation(tr); a, b = perm[:int(0.85 * len(perm))], perm[int(0.85 * len(perm)):]
    net = Head(X.shape[1], zz is not None); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=0.05); best = (1e9, None)
    for ep in range(300):
        net.train(); bi = r.permutation(a)
        for s0 in range(0, len(bi), 32):
            j = bi[s0:s0 + 32]; loss = ((net(xt[j], None if zt is None else zt[j]) - yt[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): l = ((net(xt[b], None if zt is None else zt[b]) - yt[b]) ** 2).mean().item()
        if l < best[0]: best = (l, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad(): return net(xt[va], None if zt is None else zt[va]).numpy() * sy + my
OUT = {a: np.zeros(len(y)) for a in 'ABC'}
for k in range(5):
    tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
    for a in 'ABC': OUT[a][va] = np.mean([run(a, tr, va, 100 * k + s) for s in range(5)], 0)
    print('fold', k, ' '.join('%s %.3f' % (a, np.sqrt(np.mean((OUT[a][va] - y[va]) ** 2))) for a in 'ABC'), flush=True)
g = np.mean([np.load(_repo_io(f'../push-600/dev-{v}-angh.npz'), allow_pickle=True)[part + '_g'] for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']], 0)
bl = lambda gg: (0.25 * p021 + 0.75 * np.where(failed, p021, gg) - y) ** 2
base = bl(g)
print('039 grain %.3f W.75 %.4f' % (np.sqrt(np.mean((g - y) ** 2)), np.sqrt(base.mean())))
for a in 'ABC':
    se = bl(0.8 * g + 0.2 * OUT[a]); d = se - base
    print('%s standalone %.3f errcorr(039) %.3f | fixed 0.8*039+0.2*head: W.75 %.4f dMSE %+.2f (se %.2f) folds better %d/5' % (a, np.sqrt(np.mean((OUT[a] - y) ** 2)), np.corrcoef(OUT[a] - y, g - y)[0, 1], np.sqrt(se.mean()), d.mean(), d.std() / np.sqrt(len(d)), sum(np.mean(d[fold == k]) < 0 for k in range(5))))
np.savez(_repo_io(f'zonehead-{part}.npz'), **OUT)
