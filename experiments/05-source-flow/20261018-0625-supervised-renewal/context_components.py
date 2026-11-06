"""Pure architecture/auxiliary definitions from the source-local context study."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
WIDTH=96
def positions():
    yy, xx = np.mgrid[:16, :16]
    freq = 1 / 10000 ** (np.arange(24) / 24)
    pe = np.c_[np.sin(xx.ravel()[:, None] * freq), np.cos(xx.ravel()[:, None] * freq), np.sin(yy.ravel()[:, None] * freq), np.cos(yy.ravel()[:, None] * freq)].astype(np.float32)
    return torch.from_numpy(np.vstack([np.zeros((1, 96), np.float32), pe]))[None]

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.out = nn.Linear(WIDTH, WIDTH)
        self.n2 = nn.LayerNorm(WIDTH)
        self.ff = nn.Sequential(nn.Linear(WIDTH, 192), nn.GELU(), nn.Dropout(0.1), nn.Linear(192, WIDTH))
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, 4, 24).permute(2, 0, 3, 1, 4).unbind(0)
        z = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0).transpose(1, 2).reshape(b, n, WIDTH)
        x = x + self.drop(self.out(z))
        return x + self.drop(self.ff(self.n2(x)))

class ContextNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.pixel = nn.Sequential(nn.Linear(64, 64), nn.GELU())
        self.merge = nn.Linear(256, WIDTH)
        self.cls = nn.Parameter(torch.randn(1, 1, WIDTH) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, WIDTH) * 0.02)
        self.register_buffer('pos', positions())
        self.blocks = nn.Sequential(*[Block() for _ in range(4)])
        self.norm = nn.LayerNorm(WIDTH)
        self.decoder = nn.Sequential(nn.Linear(WIDTH, 64), nn.GELU(), nn.Linear(64, 16))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x, perm, mask=None, decode=True):
        b = len(x)
        p = x.unfold(2, 8, 8).unfold(3, 8, 8).squeeze(1).reshape(b, 32, 32, 64)
        p = self.pixel(p).reshape(b, 16, 2, 16, 2, 64).permute(0, 1, 3, 2, 4, 5).reshape(b, 256, 256)
        z = self.merge(p)
        z = torch.gather(z, 1, perm[:, :, None].expand(-1, -1, WIDTH))
        if mask is not None:
            z = torch.where(mask[:, :, None], self.mask_token, z)
        z = torch.cat([self.cls.expand(b, -1, -1), z], 1) + self.pos
        z = self.norm(self.blocks(z))
        v = z[:, 1:]
        desc = torch.cat([z[:, 0], v.mean(1), (v.var(1, correction=0) + 1e-06).sqrt()], 1)
        return (self.decoder(v) if decode else None, desc)

def targets(x, perm):
    b = len(x)
    t = F.avg_pool2d(x, 4).reshape(b, 1, 16, 4, 16, 4).permute(0, 2, 4, 1, 3, 5).reshape(b, 256, 16)
    return torch.gather(t, 1, perm[:, :, None].expand(-1, -1, 16))

def losses(pred, t, mask, per_image=False):
    pm = pred.mean(-1, keepdim=True)
    tm = t.mean(-1, keepdim=True)
    pa = (pred - pm).reshape(-1, 256, 4, 4)
    ta = (t - tm).reshape(-1, 256, 4, 4)
    dc = (pm - tm).square().squeeze(-1)
    ac = (pa - ta).square().mean((-2, -1))
    grad = 0.5 * ((torch.diff(pa, dim=-1) - torch.diff(ta, dim=-1)).square().mean((-2, -1)) + (torch.diff(pa, dim=-2) - torch.diff(ta, dim=-2)).square().mean((-2, -1)))
    values = torch.stack([(x * mask).sum(1) / mask.sum(1) for x in [dc, ac, grad]], 1)
    return values if per_image else (values.mean(0) * torch.tensor([1.0, 4.0, 2.0], device=pred.device)).sum()

def mask_batch(n, rng):
    m = np.zeros((n, 64), bool)
    for row in m:
        row[rng.choice(64, 26, replace=False)] = True
    return m.reshape(n, 8, 8).repeat(2, 1).repeat(2, 2).reshape(n, 256)

def normalize(p, meta):
    return (p.astype(np.float32) / 255 - meta['image_mean']) / meta['image_std']
