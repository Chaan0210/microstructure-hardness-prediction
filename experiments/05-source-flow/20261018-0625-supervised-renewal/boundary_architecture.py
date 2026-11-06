"""Architecture-only copy of existing boundary encoder, without side effects."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU())

def pos():
    y, x = np.mgrid[:16, :16]
    freq = 1 / 10000 ** (np.arange(24) / 24)
    p = np.c_[np.sin(x.ravel()[:, None] * freq), np.cos(x.ravel()[:, None] * freq), np.sin(y.ravel()[:, None] * freq), np.cos(y.ravel()[:, None] * freq)]
    return torch.from_numpy(p.astype(np.float32))[None]

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(96)
        self.qkv = nn.Linear(96, 288)
        self.out = nn.Linear(96, 96)
        self.n2 = nn.LayerNorm(96)
        self.ff = nn.Sequential(nn.Linear(96, 192), nn.GELU(), nn.Dropout(0.1), nn.Linear(192, 96))
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, 4, 24).permute(2, 0, 3, 1, 4).unbind(0)
        z = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).transpose(1, 2).reshape(b, n, 96)
        x = x + self.drop(self.out(z))
        return x + self.drop(self.ff(self.n2(x)))

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.s1 = conv(1, 16)
        self.s2 = conv(16, 32)
        self.s3 = conv(32, 64)
        self.s4 = conv(64, 96)
        self.register_buffer('position', pos())
        self.attention = nn.Sequential(Block(), Block())
        self.norm = nn.LayerNorm(96)
        self.d3 = conv(160, 64)
        self.d2 = conv(96, 32)
        self.d1 = conv(48, 16)
        self.boundary = nn.Conv2d(16, 1, 1)
        self.affinity = nn.Conv2d(64, 2, 3, padding=1)
        self.head = nn.Sequential(nn.Linear(320, 64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        a = self.s1(F.avg_pool2d(x, 2))
        b = self.s2(F.avg_pool2d(a, 2))
        d = self.s3(F.avg_pool2d(b, 2))
        h = self.s4(F.avg_pool2d(d, 2))
        z = self.norm(self.attention(h.flatten(2).transpose(1, 2) + self.position)).transpose(1, 2).reshape(-1, 96, 16, 16)
        d = self.d3(torch.cat([F.interpolate(z, scale_factor=2, mode='bilinear', align_corners=False), d], 1))
        b = self.d2(torch.cat([F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False), a], 1))
        f = torch.cat([z.mean((2, 3)), (z.var((2, 3), correction=0) + 1e-06).sqrt(), d.mean((2, 3)), (d.var((2, 3), correction=0) + 1e-06).sqrt()], 1)
        q = self.affinity(d)
        edges = torch.cat([q[:, 0, :, :-1].flatten(1), q[:, 1, :-1, :].flatten(1)], 1)
        return (self.boundary(a)[:, 0], edges, f, self.head(f)[:, 0])
