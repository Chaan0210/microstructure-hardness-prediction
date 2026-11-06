"""[x2 variant] Same as ft_cellpose.py but at 2x scale: degrade at native resolution, upsample image 2x (bicubic) and target masks 2x
(nearest), train on random 256x256 crops (each covers 128x128 native px). Inference must segment 2x-upsampled images (seg_ft.py --x2).
Fine-tune Cellpose-SAM to segment degraded images: clean train images + random blur/contrast/noise degradation as input,
the zero-shot masks-v3 of the clean original as target. Label-free (no hardness used); test images never touched.
usage: ft_cellpose.py NAME N_STEPS [holdout80=1] [lr=1e-5] [batch=4] [freeze=0] [n_clean=250] [dmode=broad|heavy]
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from cellpose import models, dynamics, transforms
from cellpose.train import _loss_fn_seg
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

name, n_steps = sys.argv[1], int(sys.argv[2])
holdout = int(sys.argv[3]) if len(sys.argv) > 3 else 1
lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
bs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
freeze = int(sys.argv[6]) if len(sys.argv) > 6 else 0
n_clean = int(sys.argv[7]) if len(sys.argv) > 7 else 250
dmode = sys.argv[8] if len(sys.argv) > 8 else 'broad'   # broad | heavy (70% of samples matched to the real noisy half: blur 0.7-1.5, contrast 0.65-0.95, noise 12-22)
P = Path(_repo_io('../..')); R = Path(_repo_io('../grain-cellpose-220'))
dev = torch.device('mps')
M = np.load(_repo_path(R, 'masks-v3-train.npz'), allow_pickle=True); ids = [str(i) for i in M['ids']]
nz = np.load(_repo_io('noise_train.npy')); order = np.argsort(nz)
tr_idx = order[80:n_clean] if holdout else order[:n_clean]   # 80 cleanest = fidelity hold-out (fidelity_exp.py src)
print(json.dumps(dict(name=name, n_train=len(tr_idx), steps=n_steps, lr=lr, bs=bs, freeze=freeze, holdout=holdout, dmode=dmode)), flush=True)

imgs = [np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png'))) for k in tr_idx]
t0 = time.time()
flows = dynamics.labels_to_flows([np.kron(M['masks'][k].astype(np.int32), np.ones((2, 2), np.int32)) for k in tr_idx], device=dev)  # 2x masks
flows = [f[1:4].astype(np.float32) for f in flows]
print(json.dumps(dict(flows_sec=round(time.time() - t0))), flush=True)

def degrade(im, rng):
    f = im.astype(np.float32)
    sb = rng.uniform(0.0, 1.6); c = rng.uniform(0.55, 1.0); sn = rng.uniform(0.0, 24.0)
    if dmode == 'heavy' and rng.rand() < 0.7: sb, c, sn = rng.uniform(0.7, 1.5), rng.uniform(0.65, 0.95), rng.uniform(12, 22)
    if rng.rand() < 0.15: sb, c, sn = 0.0, 1.0, rng.uniform(0, 4)          # keep some near-clean inputs
    g = ndi.gaussian_filter(f, sb) if sb > 0.05 else f
    m = g.mean(); g = m + c * (g - m) + rng.uniform(-15, 15)
    g = g + rng.normal(0, sn, f.shape)
    return np.clip(np.round(g), 0, 255).astype(np.uint8)

def d4(a, fl, t, ud, lr_):
    """a: HxW image, fl: [binary, fy, fx]; apply transpose / flipud / fliplr consistently to flows."""
    b, fy, fx = fl
    if t: a, b, fy, fx = a.T, b.T, fx.T, fy.T
    if ud: a, b, fy, fx = a[::-1], b[::-1], -fy[::-1], fx[::-1]
    if lr_: a, b, fy, fx = a[:, ::-1], b[:, ::-1], fy[:, ::-1], -fx[:, ::-1]
    return np.ascontiguousarray(a), np.ascontiguousarray(np.stack([b, fy, fx]))

def norm(u8):
    x = u8.astype(np.float32)[..., None]
    return transforms.normalize_img(x, normalize=True, axis=-1)[..., 0]

model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(R, 'weights/cpsam')), use_bfloat16=False)
net = model.net; net.train()
if freeze > 0:
    for p in net.encoder.patch_embed.parameters(): p.requires_grad = False
    net.encoder.pos_embed.requires_grad = False
    for blk in net.encoder.blocks[:freeze]:
        for p in blk.parameters(): p.requires_grad = False
def fwd(x):
    """CPSAM.forward with per-block gradient checkpointing (same stochastic depth as the original training mode)."""
    x = F.conv2d(x, net.encoder.patch_embed.proj.weight.data[:, :x.shape[1]], bias=net.encoder.patch_embed.proj.bias.data, stride=net.ps)
    x = x.permute(0, 2, 3, 1) + net.encoder.pos_embed
    nlay = len(net.encoder.blocks)
    rd = (torch.rand((len(x), nlay), device=x.device) < torch.linspace(0, net.rdrop, nlay, device=x.device)).to(x.dtype)
    for i, blk in enumerate(net.encoder.blocks):
        m = rd[:, i].view(-1, 1, 1, 1)
        h = checkpoint(blk, x, use_reentrant=False) if any(p.requires_grad for p in blk.parameters()) else blk(x)
        x = x * m + h * (1 - m)
    x = net.encoder.neck(x.permute(0, 3, 1, 2))
    return F.conv_transpose2d(net.out(x), net.W2, stride=net.ps, padding=0)
params = [p for p in net.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.1)
warm = max(1, n_steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + np.cos(np.pi * min(s, n_steps) / n_steps)) * 0.95 + 0.05))
rng = np.random.RandomState(0); t0 = time.time(); la = []
for step in range(n_steps):
    xs, ls = [], []
    for k in rng.randint(0, len(imgs), bs):
        d = degrade(imgs[k], rng)                                            # degrade at native resolution
        up = np.asarray(Image.fromarray(d).resize((512, 512), Image.BICUBIC))  # then 2x, as at inference
        a, fl = d4(up, flows[k], rng.rand() < 0.5, rng.rand() < 0.5, rng.rand() < 0.5)
        oy, ox = rng.randint(0, 257, 2)
        xs.append(np.ascontiguousarray(norm(np.ascontiguousarray(a))[oy:oy + 256, ox:ox + 256]))   # normalise the full 2x image, as cellpose eval does
        ls.append(np.ascontiguousarray(fl[:, oy:oy + 256, ox:ox + 256]))
    X = torch.from_numpy(np.stack(xs)[:, None]).to(dev); L = torch.from_numpy(np.stack(ls)).to(dev)
    y = fwd(X)
    loss = _loss_fn_seg(L, y, dev)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step(); la.append(float(loss))
    if (step + 1) % 25 == 0:
        print(json.dumps(dict(step=step + 1, loss=round(float(np.mean(la[-25:])), 4), lr=f'{sched.get_last_lr()[0]:.2e}', sec=round(time.time() - t0))), flush=True)
    if (step + 1) % 250 == 0 or step + 1 == n_steps:
        net.save_model(f'ftx2-{name}.pt')
        if (step + 1) % 250 == 0: net.save_model(f'ftx2-{name}s{step + 1}.pt')   # step snapshot (ft-a1s250.pt = step 250 of the 800-step a1 run)
print('DONE', flush=True)
