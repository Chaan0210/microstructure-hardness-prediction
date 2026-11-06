"""Pure definitions copied from the existing phase/geometry U-Net; no I/O."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
import torch
from torch import nn
from torch.nn import functional as F
def targets(a, m):
    m = m.astype(np.int64)
    sm = ndi.gaussian_filter(a / 255.0, 1.0)
    inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
    props = {p.label: p for p in measure.regionprops(m)}
    ipr = {p.label: p for p in measure.regionprops(inner, intensity_image=sm)}
    labs = np.array([l for l in props if l in ipr and ipr[l].area >= 3])
    A = np.array([props[l].area for l in labs], float)
    I = np.array([ipr[l].intensity_mean for l in labs])
    AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1.0) for l in labs])
    SOL = np.array([props[l].solidity for l in labs])
    TH = np.array([props[l].orientation for l in labs])
    o = np.argsort(I)
    c = np.cumsum(A[o]) / A.sum()
    hi = I[o][np.searchsorted(c, 0.75)]
    lo = np.percentile(sm, 0.1)
    z = (hi - I) / max(hi - lo, 0.001)
    pore = (z > 0.75) & (SOL > 0.85) & (AR < 2.5) & (A < 3.0 * np.median(A))
    ph = ~pore & (z > 0.3)
    lut = np.zeros(m.max() + 1, np.int64)
    lut[labs[ph]] = 1
    lut[labs[pore]] = 2
    T = lut[m]
    bnd = (m == 0) | (ndi.minimum_filter(m, 3) != ndi.maximum_filter(m, 3))
    T[bnd & (T != 2)] = 3
    la = np.zeros(m.max() + 1, np.float32)
    la[labs] = np.log(A)
    lar = np.log(np.maximum(AR, 1.0))
    c2 = np.zeros(m.max() + 1, np.float32)
    s2 = np.zeros(m.max() + 1, np.float32)
    c2[labs] = lar * np.cos(2 * TH)
    s2[labs] = lar * np.sin(2 * TH)
    valid = np.zeros(m.max() + 1, bool)
    valid[labs[~pore]] = True
    zz = np.zeros(m.max() + 1, np.float32)
    zz[labs] = np.clip(z, -0.3, 1.0)
    R = np.stack([la[m], c2[m], s2[m], zz[m]])
    V = valid[m] & ~bnd
    return (T.astype(np.int64), R.astype(np.float32), V)

def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class UNet(nn.Module):

    def __init__(s, w=32, nout=8):
        super().__init__()
        ch = (w, 2 * w, 4 * w, 8 * w)
        s.e = nn.ModuleList()
        c0 = 1
        for c in ch:
            s.e.append(blk(c0, c))
            c0 = c
        s.u = nn.ModuleList()
        s.d = nn.ModuleList()
        for c in reversed(ch[:-1]):
            s.u.append(nn.ConvTranspose2d(c0, c, 2, stride=2))
            s.d.append(blk(2 * c, c))
            c0 = c
        s.out = nn.Conv2d(c0, nout, 1)

    def forward(s, x):
        sk = []
        for i, b in enumerate(s.e):
            x = b(x)
            if i < len(s.e) - 1:
                sk.append(x)
                x = F.max_pool2d(x, 2)
        for u, d in zip(s.u, s.d):
            x = u(x)
            x = d(torch.cat([x, sk.pop()], 1))
        return s.out(x)
