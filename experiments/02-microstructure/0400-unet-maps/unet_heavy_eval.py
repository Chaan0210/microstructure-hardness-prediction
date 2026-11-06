"""Label-free U-Net selection metric: phase-fraction (and logA) recovery on HEAVILY degraded held-out clean train images
(the 30 'val' images of phase_unet*.py seed 0). usage: unet_heavy_eval.py KIND:NAME [...]  KIND in u (phase_unet.py), m (unet2), z (unet3)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, torch
from pathlib import Path
from scipy import ndimage as ndi
HERE = Path(__file__).resolve().parent
import importlib
sys.argv_models = sys.argv[1:]; sys.argv = ['phase_unet2.py', 'evalonly', '1', '0', '0', '32']
src = open(_repo_path(HERE, 'phase_unet2.py')).read(); exec(src.split("if __name__ == '__main__':")[0])   # builds targets for clean imgs, UNet, degrade, norm
import torch.nn as nn
def blk1(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
r = np.random.RandomState(999); XS = []; TP = []
def heavy(a, lab, r):
    f = a.copy(); off = r.normal(0, r.uniform(3, 7), lab.max() + 1); off[0] = 0; f = f + off[lab]
    cq = r.normal(0, 1, 5); sh = cq[0] * yy + cq[1] * xx + cq[2] * yy * yy + cq[3] * xx * xx + cq[4] * xx * yy; f = f + r.uniform(0, 8) * sh / max(sh.std(), 1e-6)
    f = ndi.gaussian_filter(f, r.uniform(1.0, 2.0)); c = r.uniform(0.6, 0.85); mu_ = f.mean(); f = mu_ + c * (f - mu_) + r.uniform(-15, 15)
    return np.clip(np.round(f + r.normal(0, r.uniform(14, 22), f.shape)), 0, 255)
for k in val:
    g = (TT[k] <= 1); tp = ((TT[k] == 1) & g).sum() / g.sum()
    for _ in range(4): XS.append(norm(heavy(imgs[k], L[k], r))); TP.append(tp)
XS = np.stack(XS); TP = np.array(TP)
def tta_probs(net, nout):
    out = []
    for i in range(0, len(XS), 8):
        xb = XS[i:i + 8]; acc = 0
        for kk in range(8):
            v = xb
            if kk & 1: v = v.transpose(0, 2, 1)
            if kk & 2: v = v[:, ::-1]
            if kk & 4: v = v[:, :, ::-1]
            with torch.no_grad(): p = net(torch.tensor(np.ascontiguousarray(v))[:, None].to(dev))[:, :4].softmax(1).cpu().numpy()
            if kk & 4: p = p[:, :, :, ::-1]
            if kk & 2: p = p[:, :, ::-1]
            if kk & 1: p = p.transpose(0, 1, 3, 2)
            acc = acc + p
        out.append(acc / 8)
    return np.concatenate(out)
for spec in sys.argv_models:
    kind, name = spec.split(':')
    if kind == 'u':
        ns = {}; s1 = open(_repo_path(HERE, 'unet_feats.py')).read(); 
        class U(nn.Module):
            def __init__(s, ch=(32, 64, 128, 256)):
                super().__init__(); s.e = nn.ModuleList(); c0 = 1
                for c in ch: s.e.append(blk1(c0, c)); c0 = c
                s.u = nn.ModuleList(); s.d = nn.ModuleList()
                for c in reversed(ch[:-1]): s.u.append(nn.ConvTranspose2d(c0, c, 2, stride=2)); s.d.append(blk1(2 * c, c)); c0 = c
                s.out = nn.Conv2d(c0, 4, 1)
            def forward(s, x):
                sk = []
                for i, b in enumerate(s.e):
                    x = b(x)
                    if i < len(s.e) - 1: sk.append(x); x = F.max_pool2d(x, 2)
                for u, d in zip(s.u, s.d): x = u(x); x = d(torch.cat([x, sk.pop()], 1))
                return s.out(x)
        net = U().to(dev); net.load_state_dict(torch.load(_repo_path(HERE, f'unet-{name}.pt'), map_location=dev))
    else:
        w = 48 if 'w48' in name else 32; net = UNet(w, 7 if kind == 'm' else 8).to(dev)
        net.load_state_dict(torch.load(_repo_path(HERE, f'unet2-{name}.pt' if kind == 'm' else f'unet3-{name}.pt'), map_location=dev))
    net.eval(); p = tta_probs(net, 4); g = p[:, 0] + p[:, 1]
    ph = p[:, 1].sum((1, 2)) / g.sum((1, 2)); am = p.argmax(1); hard = (am == 1).sum((1, 2)) / np.maximum(((am == 0) | (am == 1)).sum((1, 2)), 1)
    print(f'{spec:12s} heavy-degradation phase recovery: soft corr {np.corrcoef(ph, TP)[0,1]:.4f} rmse {np.sqrt(np.mean((ph-TP)**2)):.4f} | hard corr {np.corrcoef(hard, TP)[0,1]:.4f} rmse {np.sqrt(np.mean((hard-TP)**2)):.4f}', flush=True)
