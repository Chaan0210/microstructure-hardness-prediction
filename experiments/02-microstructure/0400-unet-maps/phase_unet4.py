"""phase_unet2.py + channel 7: darkness z of the containing grain (features.py intensity rule, continuous). Multi-task label-free pixel U-Net: degraded clean TRAIN image -> (a) class {matrix, phase, pore, boundary}, (b) log area of the
containing grain, (c) shape-tensor vector (logAR*cos2t, logAR*sin2t) of the containing grain. Targets from zero-shot masks-v3 of clean
train images (noise < train median) + features.py intensity rule. No hardness labels, no test images.
usage: phase_unet4.py NAME ITERS [seed=0] [use_val_for_train=0] [width=32]"""
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
allv = int(sys.argv[4]) if len(sys.argv) > 4 else 0; width = int(sys.argv[5]) if len(sys.argv) > 5 else 32
torch.manual_seed(seed); rng = np.random.RandomState(seed); dev = torch.device('mps')
M = np.load(_repo_path(HERE.parent, 'grain-cellpose-220/masks-v3-train.npz')); ids = [str(i) for i in M['ids']]
nz = np.load(_repo_path(HERE.parent, 'push-300/noise_train.npy')); clean = np.where(nz < np.median(nz))[0]
vr0 = np.random.RandomState(0); val = clean[vr0.permutation(len(clean))[:30]]
SRCQ = float(__import__('os').environ.get('SRCQ', '0.5'))   # source images: noise below this train quantile (targets from their zero-shot masks)
src_set = np.where(nz < np.quantile(nz, SRCQ))[0] if SRCQ != 0.5 else clean
trn = src_set if allv else np.setdiff1d(src_set, val); clean = np.union1d(src_set, val)

def targets(a, m):
    m = m.astype(np.int64); sm = ndi.gaussian_filter(a / 255.0, 1.0)
    inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}; ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = np.array([l for l in props if l in ipr and ipr[l].area >= 3])
    A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1.0) for l in labs]); SOL = np.array([props[l].solidity for l in labs])
    TH = np.array([props[l].orientation for l in labs])
    o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3.0 * np.median(A)); ph = ~pore & (z > 0.3)
    lut = np.zeros(m.max() + 1, np.int64); lut[labs[ph]] = 1; lut[labs[pore]] = 2
    T = lut[m]; bnd = (m == 0) | (ndi.minimum_filter(m, 3) != ndi.maximum_filter(m, 3)); T[bnd & (T != 2)] = 3
    la = np.zeros(m.max() + 1, np.float32); la[labs] = np.log(A); lar = np.log(np.maximum(AR, 1.0))
    # skimage orientation: angle between row axis and major axis; use doubled angle (sign-free)
    c2 = np.zeros(m.max() + 1, np.float32); s2 = np.zeros(m.max() + 1, np.float32); c2[labs] = lar * np.cos(2 * TH); s2[labs] = lar * np.sin(2 * TH)
    valid = np.zeros(m.max() + 1, bool); valid[labs[~pore]] = True
    zz = np.zeros(m.max() + 1, np.float32); zz[labs] = np.clip(z, -0.3, 1.0)
    R = np.stack([la[m], c2[m], s2[m], zz[m]]); V = valid[m] & ~bnd
    return T.astype(np.int64), R.astype(np.float32), V

imgs = {k: np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png'))).astype(np.float32) for k in clean}
TT = {}; RR = {}; VV = {}; L = {k: M['masks'][k].astype(np.int64) for k in clean}
for k in clean: TT[k], RR[k], VV[k] = targets(imgs[k], M['masks'][k])
cls_freq = np.bincount(np.concatenate([TT[k].ravel() for k in trn]), minlength=4) / (len(trn) * 65536)
la_mu = float(np.mean([RR[k][0][VV[k]].mean() for k in trn])); la_sd = float(np.mean([RR[k][0][VV[k]].std() for k in trn]) + 0.5)
print(json.dumps(dict(name=name, n_train=len(trn), n_val=len(val), cls_freq=cls_freq.round(4).tolist(), la_mu=la_mu, la_sd=la_sd, width=width)), flush=True)
yy, xx = np.mgrid[0:256, 0:256] / 255.0 - 0.5

HEAVY = float(__import__('os').environ.get('HEAVY', '0'))
OFFMAX = float(__import__('os').environ.get('OFFMAX', '7.0'))   # max per-grain intensity offset SD in the default branch
HOFF_LO, HOFF_HI = float(__import__('os').environ.get('HOFF_LO', '3')), float(__import__('os').environ.get('HOFF_HI', '7'))   # heavy-branch per-grain offset SD range
def degrade(a, lab, r):
    f = a.copy()
    if r.rand() < HEAVY:
        off = r.normal(0, r.uniform(HOFF_LO, HOFF_HI), lab.max() + 1); off[0] = 0; f = f + off[lab]
        cq = r.normal(0, 1, 5); sh = cq[0] * yy + cq[1] * xx + cq[2] * yy * yy + cq[3] * xx * xx + cq[4] * xx * yy; f = f + r.uniform(0, 8) * sh / max(sh.std(), 1e-6)
        f = ndi.gaussian_filter(f, r.uniform(0.8, 2.2)); c = r.uniform(0.55, 0.9); mu_ = f.mean(); f = mu_ + c * (f - mu_) + r.uniform(-15, 15)
        return np.clip(np.round(f + r.normal(0, r.uniform(12, 24), f.shape)), 0, 255)
    if r.rand() < 0.85:
        sg = r.uniform(0, OFFMAX); off = r.normal(0, sg, lab.max() + 1); off[0] = 0; f = f + off[lab]
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

def d4(x, t, R, V, k):
    """transpose / flipud / fliplr; the doubled-angle vector (c2, s2) transforms: transpose -> s2 stays? use explicit rules."""
    c2, s2 = R[1], R[2]
    if k & 1: x, t, V = x.T, t.T, V.T; R = R.transpose(0, 2, 1); c2, s2 = -R[1], R[2]   # swap row/col axes: theta -> pi/2 - theta
    if k & 2: x, t, V = x[::-1], t[::-1], V[::-1]; R = R[:, ::-1]; c2, s2 = R[1] if not (k & 1) else c2[::-1], -(R[2] if not (k & 1) else s2[::-1])
    if k & 4: x, t, V = x[:, ::-1], t[:, ::-1], V[:, ::-1]; R = R[:, :, ::-1]; c2, s2 = (R[1] if not (k & 3) else c2[:, ::-1]), -(R[2] if not (k & 3) else s2[:, ::-1])
    out = np.stack([R[0], c2 if (k & 7) else R[1], s2 if (k & 7) else R[2], R[3]])
    return np.ascontiguousarray(x), np.ascontiguousarray(t), np.ascontiguousarray(out), np.ascontiguousarray(V)

def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
class UNet(nn.Module):
    def __init__(s, w=32, nout=8):
        super().__init__(); ch = (w, 2 * w, 4 * w, 8 * w); s.e = nn.ModuleList(); c0 = 1
        for c in ch: s.e.append(blk(c0, c)); c0 = c
        s.u = nn.ModuleList(); s.d = nn.ModuleList()
        for c in reversed(ch[:-1]): s.u.append(nn.ConvTranspose2d(c0, c, 2, stride=2)); s.d.append(blk(2 * c, c)); c0 = c
        s.out = nn.Conv2d(c0, nout, 1)
    def forward(s, x):
        sk = []
        for i, b in enumerate(s.e):
            x = b(x)
            if i < len(s.e) - 1: sk.append(x); x = F.max_pool2d(x, 2)
        for u, d in zip(s.u, s.d): x = u(x); x = d(torch.cat([x, sk.pop()], 1))
        return s.out(x)

if __name__ == '__main__':
    net = UNet(width).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=iters, pct_start=0.1)
    w = torch.tensor((1.0 / np.sqrt(cls_freq + 1e-3)) / (1.0 / np.sqrt(cls_freq + 1e-3)).mean(), dtype=torch.float32, device=dev)
    sc = torch.tensor([la_sd, 0.3, 0.3, 0.15], device=dev)[None, :, None, None]; mu = torch.tensor([la_mu, 0.0, 0.0, 0.15], device=dev)[None, :, None, None]
    vr = np.random.RandomState(123); VX = []; VT = []; VR = []; VM = []
    for k in val:
        for _ in range(2): VX.append(norm(degrade(imgs[k], L[k], vr))); VT.append(TT[k]); VR.append(RR[k]); VM.append(VV[k])
    VX = torch.tensor(np.stack(VX))[:, None].to(dev); VT = torch.tensor(np.stack(VT)).to(dev); VR = torch.tensor(np.stack(VR)).to(dev); VM = torch.tensor(np.stack(VM)).to(dev)
    t0 = time.time(); bs = 8
    for it in range(1, iters + 1):
        net.train(); ks = rng.choice(trn, bs); xb = []; tb = []; rb = []; vb = []
        for k in ks:
            x, t, R, V = d4(norm(degrade(imgs[k], L[k], rng)), TT[k], RR[k], VV[k], rng.randint(8)); xb.append(x); tb.append(t); rb.append(R); vb.append(V)
        xb = torch.tensor(np.stack(xb))[:, None].to(dev); tb = torch.tensor(np.stack(tb)).to(dev); rb = torch.tensor(np.stack(rb)).to(dev); vb = torch.tensor(np.stack(vb)).to(dev)
        out = net(xb); ce = F.cross_entropy(out[:, :4], tb, weight=w)
        reg = (((out[:, 4:] - (rb - mu) / sc) ** 2).mean(1) * vb).sum() / vb.sum().clamp(min=1)
        loss = ce + 0.5 * reg; opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0 or it == iters:
            net.eval()
            with torch.no_grad():
                o = torch.cat([net(VX[i:i + 10]) for i in range(0, len(VX), 10)]); pr = o[:, :4].softmax(1); rg = o[:, 4:] * sc + mu
                g = (VT <= 1); ph_true = ((VT == 1) & g).float().sum((1, 2)) / g.float().sum((1, 2)).clamp(min=1)
                ph_pred = pr[:, 1].sum((1, 2)) / (pr[:, 0] + pr[:, 1]).sum((1, 2)).clamp(min=1)
                cc = np.corrcoef(ph_true.cpu().numpy(), ph_pred.cpu().numpy())[0, 1]
                la_t = (VR[:, 0] * VM).sum((1, 2)) / VM.sum((1, 2)); la_p = (rg[:, 0] * VM).sum((1, 2)) / VM.sum((1, 2)); cl = np.corrcoef(la_t.cpu().numpy(), la_p.cpu().numpy())[0, 1]
                el_t = torch.sqrt(((VR[:, 1] * VM).sum((1, 2)) / VM.sum((1, 2))) ** 2 + ((VR[:, 2] * VM).sum((1, 2)) / VM.sum((1, 2))) ** 2)
                el_p = torch.sqrt(((rg[:, 1] * VM).sum((1, 2)) / VM.sum((1, 2))) ** 2 + ((rg[:, 2] * VM).sum((1, 2)) / VM.sum((1, 2))) ** 2); ce_ = np.corrcoef(el_t.cpu().numpy(), el_p.cpu().numpy())[0, 1]
            print(json.dumps(dict(it=it, ce=round(ce.item(), 4), reg=round(reg.item(), 4), val_phase_corr=round(float(cc), 4), val_logA_corr=round(float(cl), 4), val_elong_corr=round(float(ce_), 4), sec=round(time.time() - t0))), flush=True)
    torch.save(net.state_dict(), _repo_path(HERE, f'unet4-{name}.pt'))
