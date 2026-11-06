"""Penultimate (32-channel) feature maps of the production phase U-Net u1 (push-400/unet-u1.pt), D4-TTA averaged in the original frame,
2x2 average-pooled to 128x128, float16 -> pen-u1-<split>.npz (ids, F [n, 32, 128, 128]) + the output-layer weights.
Linear readout of these maps reproduces u1's logits (1x1 conv commutes with pooling and with the D4 average); sanity: u_ph from pooled maps vs
production u_ph. usage: unet_pen.py SPLIT"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]; split = sys.argv[1]
def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
class UNet(nn.Module):
    def __init__(s, ch=(32, 64, 128, 256)):
        super().__init__(); s.e = nn.ModuleList(); c0 = 1
        for c in ch: s.e.append(blk(c0, c)); c0 = c
        s.u = nn.ModuleList(); s.d = nn.ModuleList()
        for c in reversed(ch[:-1]): s.u.append(nn.ConvTranspose2d(c0, c, 2, stride=2)); s.d.append(blk(2 * c, c)); c0 = c
        s.out = nn.Conv2d(c0, 4, 1)
    def pen(s, x):
        sk = []
        for i, b in enumerate(s.e):
            x = b(x)
            if i < len(s.e) - 1: sk.append(x); x = F.max_pool2d(x, 2)
        for u, d in zip(s.u, s.d): x = u(x); x = d(torch.cat([x, sk.pop()], 1))
        return x
def norm(f):
    lo, med, hi = np.percentile(f, [5, 50, 95]); return ((f - med) / max(hi - lo, 1.0)).astype(np.float32)
dev = torch.device('mps'); net = UNet().to(dev); net.load_state_dict(torch.load(_repo_path(E, 'push-400/unet-u1.pt'), map_location=dev)); net.eval()
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data', 'train.csv' if split == 'train' else 'sample_submission.csv')))]
Fm = np.zeros((len(ids), 32, 128, 128), np.float16); Wout = net.out.weight.detach().cpu().numpy()[:, :, 0, 0]; bout = net.out.bias.detach().cpu().numpy()
uph = np.zeros(len(ids)); uph_full = np.zeros(len(ids))
for n, i in enumerate(ids):
    a = norm(np.array(Image.open(_repo_path(P0, f'data/{split}/{i}.png'))).astype(np.float32)); views = []
    for k in range(8):
        v = a
        if k & 1: v = v.T
        if k & 2: v = v[::-1]
        if k & 4: v = v[:, ::-1]
        views.append(np.ascontiguousarray(v))
    with torch.no_grad():
        h = net.pen(torch.tensor(np.stack(views))[:, None].to(dev)); lg = net.out(h).softmax(1).cpu().numpy(); h = h.cpu().numpy()
    acc = np.zeros((32, 256, 256), np.float32); pacc = np.zeros((4, 256, 256))
    for k in range(8):
        x, p = h[k], lg[k]
        if k & 4: x, p = x[:, :, ::-1], p[:, :, ::-1]
        if k & 2: x, p = x[:, ::-1], p[:, ::-1]
        if k & 1: x, p = x.transpose(0, 2, 1), p.transpose(0, 2, 1)
        acc += x; pacc += p
    acc /= 8; pacc /= 8; pool = acc.reshape(32, 128, 2, 128, 2).mean((2, 4)); Fm[n] = pool
    uph_full[n] = pacc[1].sum() / max((pacc[0] + pacc[1]).sum(), 1)
    z = np.einsum('kc,chw->khw', Wout, pool) + bout[:, None, None]; z = np.exp(z - z.max(0)); z /= z.sum(0); uph[n] = z[1].sum() / max((z[0] + z[1]).sum(), 1)
    if (n + 1) % 100 == 0: print(n + 1, flush=True)
U = np.load(_repo_path(E, f'push-400/unet-u1-{split}.npz'), allow_pickle=True); ref = U['X'][:, list(U['keys']).index('u_ph')]
print('corr pooled-readout u_ph vs production', np.corrcoef(uph, ref)[0, 1].round(5), ' full-res recompute vs production', np.corrcoef(uph_full, ref)[0, 1].round(5))
np.savez(_repo_path(H, f'pen-u1-{split}.npz'), ids=np.array(ids), F=Fm, W=Wout, b=bout, uph_pooled=uph)
