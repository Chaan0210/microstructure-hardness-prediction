"""Apply a trained pixel U-Net (phase_unet.py) with D4 TTA to all images of a split; save features (+ float16 prob maps).
usage: unet_feats.py NAME SPLIT"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
sys.argv_saved = list(sys.argv)
HERE = _repo_path(Path(__file__).resolve().parents[1], 'push-400'); P = _RepoPath(__file__).resolve().parents[3]; SET = sys.argv[3] if len(sys.argv) > 3 else 'train_deg3'
name, split = sys.argv[1], sys.argv[2]
import importlib.util
spec = importlib.util.spec_from_file_location('pu', _repo_path(HERE, 'phase_unet.py'))
src = open(_repo_path(HERE, 'phase_unet.py')).read().split("if __name__ == '__main__':")[0]
# only need UNet + norm definitions (avoid rebuilding training targets)
ns = {}
exec(compile('\n'.join([l for l in src.split('\n')]), 'phase_unet_defs', 'exec'), ns) if False else None
import torch.nn as nn, torch.nn.functional as F
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
def norm(f):
    lo, med, hi = np.percentile(f, [5, 50, 95]); return ((f - med) / max(hi - lo, 1.0)).astype(np.float32)
dev = torch.device('mps'); net = UNet().to(dev); net.load_state_dict(torch.load(_repo_path(HERE, f'unet-{name}.pt'), map_location=dev)); net.eval()
if split == 'synth':
    ids = sorted(json.load(open(_repo_path(HERE, 'synth/' + SET + '_params.json'))).keys()); IMG = lambda i: _repo_path(HERE, f'synth/{SET}/{i}.png')
else:
    ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data', 'train.csv' if split == 'train' else 'sample_submission.csv')))]; IMG = lambda i: _repo_path(P, f'data/{split}/{i}.png')
KEYS = ['u_ph', 'u_ph_hard', 'u_pore', 'u_bnd', 'u_logbnd', 'u_ph_f32', 'u_bnd_f16', 'u_bnd_f32', 'u_bnd_rng32', 'u_ph_ent']
def tta(x):
    outs = []
    for k in range(8):
        v = x
        if k & 1: v = v.T
        if k & 2: v = v[::-1]
        if k & 4: v = v[:, ::-1]
        outs.append((k, np.ascontiguousarray(v)))
    xb = torch.tensor(np.stack([v for _, v in outs]))[:, None].to(dev)
    with torch.no_grad(): pr = net(xb).softmax(1).cpu().numpy()
    acc = np.zeros((4, 256, 256))
    for (k, _), p in zip(outs, pr):
        if k & 4: p = p[:, :, ::-1]
        if k & 2: p = p[:, ::-1]
        if k & 1: p = p.transpose(0, 2, 1)
        acc += p
    return acc / 8
X = []; maps = np.zeros((len(ids), 4, 128, 128), np.float16)
for n, i in enumerate(ids):
    a = np.array(Image.open(_repo_io(IMG(i)))).astype(np.float32); p = tta(norm(a)); maps[n] = p[:, ::2, ::2]
    g = p[0] + p[1]; o = {}
    o['u_ph'] = p[1].sum() / max(g.sum(), 1); am = p.argmax(0); o['u_ph_hard'] = (am == 1).sum() / max(((am == 0) | (am == 1)).sum(), 1)
    o['u_pore'] = p[2].mean(); o['u_bnd'] = p[3].mean(); o['u_logbnd'] = np.log(max(p[3].mean(), 1e-4))
    f = ndi.gaussian_filter(p[1], 32) / np.maximum(ndi.gaussian_filter(g, 32), 1e-3); o['u_ph_f32'] = f.std()
    for s in (16, 32):
        b = np.log(np.maximum(ndi.gaussian_filter(p[3], s), 1e-4)); o[f'u_bnd_f{s}'] = b.std()
        if s == 32: o['u_bnd_rng32'] = np.percentile(b, 95) - np.percentile(b, 5)
    q = p[1] / np.maximum(g, 1e-3); q = np.clip(q[g > 0.5], 1e-4, 1 - 1e-4); o['u_ph_ent'] = float(np.mean(-(q * np.log(q) + (1 - q) * np.log(1 - q)))) if q.size else np.nan
    X.append([float(o[k]) for k in KEYS])
    if (n + 1) % 200 == 0: print(n + 1, flush=True)
np.savez(_repo_path(HERE, f'unet-{name}-{split}{SET if split=="synth" else ""}.npz'), ids=np.array(ids), X=np.array(X), keys=np.array(KEYS))
np.savez_compressed(_repo_path(HERE, f'unetmaps-{name}-{split}{SET if split=="synth" else ""}.npz'), ids=np.array(ids), maps=maps)
print('done', np.array(X).shape)
