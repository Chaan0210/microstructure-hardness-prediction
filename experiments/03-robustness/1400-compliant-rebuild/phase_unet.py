"""[push-1400 compliant copy of push-400/phase_unet.py: degradation uses only global/mask-free transforms]
Label-free pixel classifier (U-Net): degraded clean TRAIN image -> {matrix, second phase, pore, boundary} map of the clean original.
Targets come from zero-shot Cellpose masks-v3 of clean train images (noise < train median) + the features.py intensity rule
(z=(hi-I)/(hi-lo); pore: z>0.75 & compact; phase: z>0.3). No hardness labels, no test images.
usage: phase_unet.py NAME ITERS [seed=0]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
HERE = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]
name, iters = sys.argv[1], int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.manual_seed(seed); rng = np.random.RandomState(seed); dev = torch.device('mps')
M = np.load(_repo_path(HERE.parent, 'grain-cellpose-220/masks-v3-train.npz')); ids = [str(i) for i in M['ids']]
nz = np.load(_repo_path(HERE.parent, 'push-300/noise_train.npy')); clean = np.where(nz < np.median(nz))[0]
val = clean[rng.permutation(len(clean))[:30]]; trn = np.setdiff1d(clean, val)

def targets(a, m):
    m = m.astype(np.int64); sm = ndi.gaussian_filter(a / 255.0, 1.0)
    inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}; ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = np.array([l for l in props if l in ipr and ipr[l].area >= 3])
    A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1.0) for l in labs]); SOL = np.array([props[l].solidity for l in labs])
    o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3.0 * np.median(A)); ph = ~pore & (z > 0.3)
    lut = np.zeros(m.max() + 1, np.int64); lut[labs[ph]] = 1; lut[labs[pore]] = 2
    T = lut[m]
    bnd = (m == 0) | (ndi.minimum_filter(m, 3) != ndi.maximum_filter(m, 3))
    T[bnd & (T != 2)] = 3
    return T.astype(np.int64)

imgs = {k: np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png'))).astype(np.float32) for k in clean}
T = {k: targets(imgs[k], M['masks'][k]) for k in clean}; L = {k: M['masks'][k].astype(np.int64) for k in clean}
cls_freq = np.bincount(np.concatenate([T[k].ravel() for k in trn]), minlength=4) / (len(trn) * 65536)
print(json.dumps(dict(name=name, n_train=len(trn), n_val=len(val), cls_freq=cls_freq.round(4).tolist())), flush=True)
yy, xx = np.mgrid[0:256, 0:256] / 255.0 - 0.5

def rfield(r):
    """Mask-free smooth random brightness field (unit SD, correlation scale 2.5-8 px): generic local brightness jitter,
    replaces the old per-grain offsets (which used the segmentation masks) so no analysis result touches the input image."""
    g = ndi.gaussian_filter(r.normal(0, 1, (256, 256)), r.uniform(2.5, 8.0)); return g / max(g.std(), 1e-6)

def degrade(a, lab, r):   # COMPLIANT: lab is not used to alter the image (targets only)
    f = a.copy()
    if r.rand() < 0.85:
        sg = r.uniform(0, 7.0); f = f + sg * rfield(r)
        amp = r.uniform(0, 8.0); cq = r.normal(0, 1, 5); sh = cq[0] * yy + cq[1] * xx + cq[2] * yy * yy + cq[3] * xx * xx + cq[4] * xx * yy
        f = f + amp * sh / max(sh.std(), 1e-6)
        sb = r.uniform(0.0, 2.2); f = ndi.gaussian_filter(f, sb) if sb > 0.05 else f
        c = r.uniform(0.6, 1.0); mu = f.mean(); f = mu + c * (f - mu) + r.uniform(-15, 15)
        f = f + r.normal(0, r.uniform(0, 24.0), f.shape)
    else:
        f = f + r.normal(0, r.uniform(0, 4.0), f.shape)
    return np.clip(np.round(f), 0, 255)

def norm(f):
    lo, med, hi = np.percentile(f, [5, 50, 95]); return ((f - med) / max(hi - lo, 1.0)).astype(np.float32)

def d4(x, t, k):
    if k & 1: x, t = x.T, t.T
    if k & 2: x, t = x[::-1], t[::-1]
    if k & 4: x, t = x[:, ::-1], t[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(t)

def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
class UNet(nn.Module):
    def __init__(s, ch=(32, 64, 128, 256)):
        super().__init__(); s.e = nn.ModuleList(); c0 = 1
        for c in ch: s.e.append(blk(c0, c)); c0 = c
        s.u = nn.ModuleList(); s.d = nn.ModuleList()
        for c in reversed(ch[:-1]): s.u.append(nn.ConvTranspose2d(c0, c, 2, stride=2)); s.d.append(blk(2 * c, c)); c0 = c
        s.out = nn.Conv2d(c0, 4, 1)
    def forward(s, x):
        sk = []
        for i, b in enumerate(s.e):
            x = b(x)
            if i < len(s.e) - 1: sk.append(x); x = F.max_pool2d(x, 2)
        for u, d in zip(s.u, s.d): x = u(x); x = d(torch.cat([x, sk.pop()], 1))
        return s.out(x)

if __name__ == '__main__':
    net = UNet().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=iters, pct_start=0.1)
    w = torch.tensor((1.0 / np.sqrt(cls_freq + 1e-3)) / (1.0 / np.sqrt(cls_freq + 1e-3)).mean(), dtype=torch.float32, device=dev)
    vr = np.random.RandomState(123); VX = []; VT = []
    for k in val:
        for _ in range(2):
            VX.append(norm(degrade(imgs[k], L[k], vr))); VT.append(T[k])
    VX = torch.tensor(np.stack(VX))[:, None].to(dev); VT = torch.tensor(np.stack(VT)).to(dev)
    t0 = time.time(); bs = 8
    for it in range(1, iters + 1):
        net.train(); ks = rng.choice(trn, bs); xb = []; tb = []
        for k in ks:
            x, t = d4(norm(degrade(imgs[k], L[k], rng)), T[k], rng.randint(8)); xb.append(x); tb.append(t)
        xb = torch.tensor(np.stack(xb))[:, None].to(dev); tb = torch.tensor(np.stack(tb)).to(dev)
        loss = F.cross_entropy(net(xb), tb, weight=w); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 250 == 0 or it == iters:
            net.eval()
            with torch.no_grad():
                pr = torch.cat([net(VX[i:i + 10]).softmax(1) for i in range(0, len(VX), 10)])
                acc = (pr.argmax(1) == VT).float().mean().item()
                g = (VT <= 1); ph_true = ((VT == 1) & g).float().sum((1, 2)) / g.float().sum((1, 2)).clamp(min=1)
                ph_pred = pr[:, 1].sum((1, 2)) / (pr[:, 0] + pr[:, 1]).sum((1, 2)).clamp(min=1)
                cc = np.corrcoef(ph_true.cpu().numpy(), ph_pred.cpu().numpy())[0, 1]
                bt = (VT == 3).float().mean((1, 2)).cpu().numpy(); bp = pr[:, 3].mean((1, 2)).cpu().numpy(); cb = np.corrcoef(bt, bp)[0, 1]
            print(json.dumps(dict(it=it, loss=round(loss.item(), 4), val_acc=round(acc, 4), val_phase_corr=round(float(cc), 4), val_bnd_corr=round(float(cb), 4), sec=round(time.time() - t0))), flush=True)
    torch.save(net.state_dict(), _repo_path(HERE, f'unet-{name}.pt'))
