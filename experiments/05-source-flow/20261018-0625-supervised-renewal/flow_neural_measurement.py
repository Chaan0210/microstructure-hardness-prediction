"""Small supervised measurement network over original-image flow features."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
HERE = Path(__file__).resolve().parent
KEYS = [f'{kind}_{j}' for kind in ['mean', 'sd', 'gray_covariance', 'tile_sd'] for j in range(8)]


def load(d):
    maps = np.load(_repo_path(HERE, 'flow-geometry/flowmaps-c4.npy'), mmap_mode='r')
    meta = np.load(_repo_path(HERE, 'flow-geometry/full-features.npz'))
    assert np.array_equal(meta['ids'], d['ids']) and maps.shape == (500, 4, 3, 128, 128)
    canonical = np.zeros((500, 4, 128, 128), np.float32)
    for rot in range(4):
        a = np.rot90(np.array(maps[:, rot], dtype=np.float32), -rot, axes=(-2, -1)).copy()
        y, x = a[:, 0].copy(), a[:, 1].copy()
        if rot == 1:
            a[:, 0], a[:, 1] = x, -y
        elif rot == 2:
            a[:, 0], a[:, 1] = -y, -x
        elif rot == 3:
            a[:, 0], a[:, 1] = -x, y
        canonical[:, :3] += a / 4
    canonical[:, :2] = np.clip(canonical[:, :2]/5, -2, 2)
    canonical[:, 2] = np.tanh(canonical[:, 2]/5)
    px = np.load(_repo_path(HERE, 'train-pixels.npy'), mmap_mode='r')[:, 0].astype(np.float32)
    canonical[:, 3] = px.reshape(500, 128, 2, 128, 2).mean((2, 4))/127.5-1
    assert np.isfinite(canonical).all()
    return dict(fields=torch.tensor(canonical))


def rotate_fields(z, rot):
    z = torch.rot90(z, rot, dims=(-2, -1))
    y, x = z[:, 0:1], z[:, 1:2]
    if rot == 1:
        y, x = -x, y
    elif rot == 2:
        y, x = -y, -x
    elif rot == 3:
        y, x = x, -y
    return torch.cat([y, x, z[:, 2:]], 1)


class Measurement(nn.Module):
    def __init__(self, data, base, train):
        super().__init__()
        self.data = data; self.base = base; self.step = 0
        self.net = nn.Sequential(nn.Conv2d(4, 12, 5, stride=2, padding=2), nn.SiLU(),
                                 nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(16, 8, 3, padding=1), nn.SiLU())
        for name, p in self.net.named_parameters():
            self.register_buffer('anchor_'+name.replace('.', '_'), p.detach().clone())
        with torch.no_grad():
            initial = self.extra(train)
            self.register_buffer('mean', initial.mean(0))
            self.register_buffer('scale', initial.std(0, correction=0).clamp(min=.01))

    def features(self, indices, rot):
        z = rotate_fields(self.data['fields'][indices], rot)
        f = self.net(z); mean = f.mean((2, 3))
        sd = torch.sqrt(f.var((2, 3), correction=0)+1e-6)
        g = F.adaptive_avg_pool2d(z[:, 3:4], f.shape[-2:]); g = g-g.mean((2, 3), keepdim=True)
        cov = (f*g).mean((2, 3))
        tile = torch.sqrt(F.adaptive_avg_pool2d(f, (4, 4)).flatten(2).var(2, correction=0)+1e-6)
        return torch.cat([mean, sd, cov, tile], 1)

    def extra(self, indices):
        rows = []
        for start in range(0, len(indices), 32):
            ix = indices[start:start+32]
            rows.append(self.features(ix, 0))
        return torch.cat(rows)

    def forward(self, indices):
        f = self.features(indices, 0)
        self.penalty = sum((p-getattr(self, 'anchor_'+name.replace('.', '_'))).square().sum()
                           for name, p in self.net.named_parameters())/sum(p.numel() for p in self.net.parameters())
        return torch.cat([self.base[indices], ((f-self.mean)/self.scale).clamp(-6, 6)], 1)
