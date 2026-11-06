"""Small end-to-end hardness readout on frozen z1 U-Net maps (128x128x8): local conv -> pooled (grain-weighted mean, std, 4x4-zone std) -> linear.
5 seeds per fold, 180-degree rotation augmentation (keeps doubled-angle channels), inner early stopping. Dev partition given; saves OOF.
usage: mapnet.py PARTITION"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, torch, torch.nn as nn, csv
part = sys.argv[1]; torch.set_num_threads(4)
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); mid = {str(i): k for k, i in enumerate(M['ids'])}
D = np.load(_repo_io('../push-600/dev-u1dmavgH2Z-angh.npz'), allow_pickle=True)
ids = [str(i) for i in D[part + '_ids']]; y = D[part + '_y']; fold = D[part + '_fold']
maps = M['maps'][[mid[i] for i in ids]].astype(np.float32)          # [N,8,128,128]: pm, pph, ppore, pbnd, logA, c2, s2, z
maps = maps.reshape(maps.shape[0], 8, 64, 2, 64, 2).mean((3, 5))
maps[:, 4] = (maps[:, 4] - 5.6) / 0.9; maps[:, 7] = (maps[:, 7] - 0.1) / 0.2; maps[:, 5:7] /= 0.5
F = np.load(_repo_io('feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); fp = {str(i): k for k, i in enumerate(F['ids'])}; keys = [str(k) for k in F['keys']]
nz = np.nan_to_num(F['X'][[fp[i] for i in ids], keys.index('noise')].astype(np.float32) * 255, nan=11.0); nzs = (nz - 11) / 5
dev = torch.device('mps'); X = torch.tensor(maps).to(dev); NZ = torch.tensor(nzs)[:, None].to(dev)
class Net(nn.Module):
    def __init__(s, h=24, d=12):
        super().__init__(); s.loc = nn.Sequential(nn.Conv2d(8, h, 3, padding=1), nn.ReLU(), nn.Conv2d(h, h, 3, padding=1, dilation=1), nn.ReLU(), nn.Conv2d(h, d, 1))
        s.head = nn.Linear(3 * d + 1, 1)
    def forward(s, x, nzv):
        h = s.loc(x); g = (x[:, 0:1] + x[:, 1:2]).clamp(min=1e-3); W = g.sum((2, 3))
        mu = (h * g).sum((2, 3)) / W; sd = torch.sqrt(((h - mu[:, :, None, None]) ** 2 * g).sum((2, 3)) / W + 1e-6)
        zm = (h * g).reshape(h.shape[0], h.shape[1], 4, 16, 4, 16).sum((3, 5)) / g.reshape(g.shape[0], 1, 4, 16, 4, 16).sum((3, 5))
        zs = zm.reshape(h.shape[0], h.shape[1], 16).std(2)
        return s.head(torch.cat([mu, sd, zs, nzv], 1))[:, 0]
def fit_predict(tr, va, seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    ymu, ysd = y[tr].mean(), y[tr].std(); yt = torch.tensor((y - ymu) / ysd, dtype=torch.float32).to(dev)
    perm = rng.permutation(tr); ntr, nes = perm[:int(0.85 * len(perm))], perm[int(0.85 * len(perm)):]
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-2); best = (1e9, None)
    for ep in range(120):
        net.train(); bi = rng.permutation(ntr)
        for b in range(0, len(bi), 32):
            j = bi[b:b + 32]; xb = X[j]
            if rng.rand() < 0.5: xb = torch.flip(xb, (2, 3))
            loss = ((net(xb, NZ[j]) - yt[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad(): l = ((net(X[nes], NZ[nes]) - yt[nes]) ** 2).mean().item()
            if np.isfinite(l) and l < best[0]: best = (l, {k: v.clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad(): p = 0.5 * (net(X[va], NZ[va]) + net(torch.flip(X[va], (2, 3)), NZ[va])).cpu().numpy()
    return p * ysd + ymu, best[2]
oof = np.zeros(len(y))
for k in range(5):
    tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]; ps = []; eps = []
    for sd in range(3): p, e = fit_predict(tr, va, 100 * k + sd); ps.append(p); eps.append(e)
    oof[va] = np.mean(ps, 0); print('fold', k, 'rmse %.3f' % np.sqrt(np.mean((oof[va] - y[va]) ** 2)), 'best epochs', eps, flush=True)
g = np.mean([np.load(_repo_io(f'../push-600/dev-{v}-angh.npz'), allow_pickle=True)[part + '_g'] for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']], 0)
p021 = D[part + '_p021']; failed = D[part + '_failed']; rm = lambda v: np.sqrt(np.mean((v - y) ** 2))
print('mapnet %.3f | 039 grain %.3f | errcorr %.3f | errcorr with 021 %.3f' % (rm(oof), rm(g), np.corrcoef(oof - y, g - y)[0, 1], np.corrcoef(oof - y, p021 - y)[0, 1]))
for a in (0, 0.1, 0.2, 0.3):
    gg = np.where(failed, p021, (1 - a) * g + a * oof); print('grain mix a=%.1f -> W0.75 blend %.4f' % (a, rm(0.25 * p021 + 0.75 * gg)))
np.save(_repo_io(f'mapnet-oof-{part}.npy'), oof)
