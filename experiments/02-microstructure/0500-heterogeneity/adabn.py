"""AdaBN (label-free domain adaptation): re-estimate BatchNorm running statistics of a U-Net on REAL noisy TRAIN images (noise >= median,
D4 views, no labels, no gradients), then write maps for all train images with the adapted net (noisy rows) / original net (others).
usage: adabn.py NET(u4z1|h1|m1) -> unet{2,4}maps-<net>ab-train.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
H = Path(__file__).resolve().parent; U = _repo_path(H.parent, 'push-400'); P = _RepoPath(__file__).resolve().parents[3]; net = sys.argv[1]
is4 = net.startswith('u4'); nm = net[2:] if is4 else net
src = open(_repo_path(U, 'phase_unet4.py' if is4 else 'phase_unet2.py')).read(); g = {}
exec('import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\n' + src[src.index('def blk'):src.index("if __name__ == '__main__':")], g)
dev = torch.device('mps')
def load():
    m = g['UNet'](32).to(dev); m.load_state_dict(torch.load(_repo_path(U, f'unet4-{nm}.pt' if is4 else f'unet2-{nm}.pt'), map_location=dev)); m.eval(); return m
orig = load(); ada = load()
LA = (5.636784553527832, 0.9691801071166992) if nm in ('m1', 'h1', 'z1') else (5.6343302726745605, 0.9692867994308472)
C = 8 if is4 else 7
def norm(f): lo, md, hi = np.percentile(f, [5, 50, 95]); return ((f - md) / max(hi - lo, 1.0)).astype(np.float32)
def views(x):
    vs = []
    for k in range(8):
        v = x
        if k & 1: v = v.T
        if k & 2: v = v[::-1]
        if k & 4: v = v[:, ::-1]
        vs.append(np.ascontiguousarray(v))
    return torch.tensor(np.stack(vs))[:, None].to(dev)
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]; nz = np.load(_repo_path(H.parent, 'push-300/noise_train.npy')); noisy = nz >= np.median(nz)
imgs = {i: np.array(Image.open(_repo_path(P, f'data/train/{i}.png'))).astype(np.float32) for i in ids}
for mod in ada.modules():
    if isinstance(mod, torch.nn.BatchNorm2d): mod.reset_running_stats(); mod.momentum = None
ada.train()
with torch.no_grad():
    for k in np.where(noisy)[0]: ada(views(norm(imgs[ids[k]])))
ada.eval()
def inv(p, k):
    p = p.copy()
    if k & 4: p = p[:, :, ::-1]; p[6] = -p[6]
    if k & 2: p = p[:, ::-1]; p[6] = -p[6]
    if k & 1: p = p.transpose(0, 2, 1); p[5] = -p[5]
    return p
maps = np.zeros((len(ids), C, 128, 128), np.float16)
with torch.no_grad():
    for n, i in enumerate(ids):
        model = ada if noisy[n] else orig
        o = model(views(norm(imgs[i]))); pr = o[:, :4].softmax(1); rg = o[:, 4:].clone(); rg[:, 0] = rg[:, 0] * LA[1] + LA[0]; rg[:, 1:3] = rg[:, 1:3] * 0.3
        if is4: rg[:, 3] = rg[:, 3] * 0.15 + 0.15
        out = torch.cat([pr, rg], 1).cpu().numpy(); acc = np.zeros((C, 256, 256))
        for k in range(8): acc += inv(out[k], k)
        maps[n] = (acc / 8)[:, ::2, ::2]
np.savez_compressed(_repo_path(U, f'unet4maps-{nm}ab-train.npz' if is4 else f'unet2maps-{nm}ab-train.npz'), ids=np.array(ids), maps=maps); print('done')
