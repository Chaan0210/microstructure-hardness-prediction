"""Learned noisy-preset degradation: unpaired CycleGAN-style translation between OUR clean train images (domain A, pixel noise < 9.5 gray)
and OUR real noisy train images (domain B, noise >= 12). Train images only (no test, no external data); outputs are transformations of
train images that keep their labels and zero-shot grain masks (structure kept by cycle consistency).
  G_AB(x, z): residual U-Net with a per-pixel N(0,1) noise channel (stochastic degradation), G_BA(y): residual U-Net restorer.
  D_A, D_B: PatchGAN (LSGAN) on random 128 crops with D4 augmentation.
  losses: GAN + 10 * L1(G_BA(G_AB(a, z)), a) + 10 * L1(blur(G_AB(G_BA(b), z')), blur(b))   (B-cycle compared after Gaussian blur: noise is random)
env LCYC (cycle weight, default 10), BLURA=1 (A-cycle also compared after the Gaussian blur, so lossy high-frequency changes are allowed), TAG (checkpoint/output suffix)
usage: cycgan.py train [iters=6000]  |  cycgan.py apply K   (K stochastic translations of every clean train image -> cyc/trans-k<K>.npz)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
H = Path(__file__).resolve().parent; P0 = _RepoPath(__file__).resolve().parents[4]
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); torch.manual_seed(20261013); np.random.seed(20261013)
def noise_gray(a):
    a = a.astype(np.float64); hh = (a[::2, ::2] - a[::2, 1::2] - a[1::2, ::2] + a[1::2, 1::2]) / 2; return float(np.median(np.abs(hh)) / .6745)
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data/train.csv')))]
IM = {i: np.asarray(Image.open(_repo_path(P0, 'data/train', f'{i}.png')), np.float32) for i in ids}
NZ = {i: noise_gray(IM[i]) for i in ids}
A_ids = [i for i in ids if NZ[i] < 9.5]; B_ids = [i for i in ids if NZ[i] >= 12]
def t(a): return torch.from_numpy((a - 128.0) / 64.0)[None]
def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.InstanceNorm2d(o, affine=True), nn.LeakyReLU(0.2), nn.Conv2d(o, o, 3, padding=1), nn.InstanceNorm2d(o, affine=True), nn.LeakyReLU(0.2))
class UNet(nn.Module):
    def __init__(s, cin, w=32):
        super().__init__(); s.e1, s.e2, s.e3 = block(cin, w), block(w, 2 * w), block(2 * w, 4 * w); s.b = block(4 * w, 4 * w)
        s.d3, s.d2, s.d1 = block(8 * w, 2 * w), block(4 * w, w), block(2 * w, w); s.out = nn.Conv2d(w, 1, 1); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(F.avg_pool2d(e1, 2)); e3 = s.e3(F.avg_pool2d(e2, 2)); b = s.b(F.avg_pool2d(e3, 2))
        u = lambda z: F.interpolate(z, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = s.d3(torch.cat([u(b), e3], 1)); d2 = s.d2(torch.cat([u(d3), e2], 1)); d1 = s.d1(torch.cat([u(d2), e1], 1)); return x[:, :1] + s.out(d1)
class Disc(nn.Module):
    def __init__(s, w=48):
        super().__init__()
        s.n = nn.Sequential(nn.Conv2d(1, w, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(w, 2 * w, 4, 2, 1), nn.InstanceNorm2d(2 * w), nn.LeakyReLU(0.2),
                            nn.Conv2d(2 * w, 4 * w, 4, 2, 1), nn.InstanceNorm2d(4 * w), nn.LeakyReLU(0.2), nn.Conv2d(4 * w, 4 * w, 3, 1, 1), nn.InstanceNorm2d(4 * w), nn.LeakyReLU(0.2), nn.Conv2d(4 * w, 1, 3, 1, 1))
    def forward(s, x): return s.n(x)
def d4(x, k):
    if k & 1: x = x.transpose(2, 3)
    if k & 2: x = x.flip(2)
    if k & 4: x = x.flip(3)
    return x
def crops(pool, n, size=128):
    out = []
    for _ in range(n):
        a = IM[pool[np.random.randint(len(pool))]]; y, x = np.random.randint(0, 256 - size + 1, 2); out.append(t(a[y:y + size, x:x + size]))
    return d4(torch.stack(out).to(dev), np.random.randint(8))
g = torch.tensor(np.outer(*(2 * [np.exp(-np.arange(-6, 7) ** 2 / (2 * 2.0 ** 2))])), dtype=torch.float32); g = (g / g.sum())[None, None].to(dev)
blur = lambda x: F.conv2d(F.pad(x, (6, 6, 6, 6), mode='reflect'), g)
import os
LCYC = float(os.environ.get('LCYC', '10')); BLURA = os.environ.get('BLURA') == '1'; TAG = os.environ.get('TAG', '')
mode = sys.argv[1]
if mode == 'train':
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 6000; print('A (clean)', len(A_ids), 'B (noisy)', len(B_ids), flush=True)
    GAB, GBA, DA, DB = UNet(2).to(dev), UNet(1).to(dev), Disc().to(dev), Disc().to(dev)
    og = torch.optim.Adam(list(GAB.parameters()) + list(GBA.parameters()), 2e-4, betas=(0.5, 0.999)); od = torch.optim.Adam(list(DA.parameters()) + list(DB.parameters()), 2e-4, betas=(0.5, 0.999))
    mse = nn.MSELoss(); t0 = time.time(); bs = 8
    for it in range(1, iters + 1):
        lr = 2e-4 * min(1.0, 2 * (1 - it / iters)); [pg.__setitem__('lr', lr) for o in (og, od) for pg in o.param_groups]
        a, b = crops(A_ids, bs), crops(B_ids, bs); z = torch.randn_like(a); z2 = torch.randn_like(b)
        fb = GAB(torch.cat([a, z], 1)); fa = GBA(b); ra = GBA(fb); rb = GAB(torch.cat([fa, z2], 1))
        pb, pa = DB(fb), DA(fa)
        lg = mse(pb, torch.ones_like(pb)) + mse(pa, torch.ones_like(pa)) + LCYC * (F.l1_loss(blur(ra), blur(a)) if BLURA else F.l1_loss(ra, a)) + LCYC * F.l1_loss(blur(rb), blur(b))
        og.zero_grad(); lg.backward(); og.step()
        qb_r, qb_f, qa_r, qa_f = DB(b), DB(fb.detach()), DA(a), DA(fa.detach())
        ld = 0.5 * (mse(qb_r, torch.ones_like(qb_r)) + mse(qb_f, torch.zeros_like(qb_f)) + mse(qa_r, torch.ones_like(qa_r)) + mse(qa_f, torch.zeros_like(qa_f)))
        od.zero_grad(); ld.backward(); od.step()
        if it % 250 == 0:
            print(f'it {it} lossG {lg.item():.3f} lossD {ld.item():.3f} cycA {F.l1_loss(ra, a).item():.4f} fake-noise {noise_gray((fb[0, 0].detach().cpu().numpy() * 64 + 128)):.2f} '
                  f'{time.time() - t0:.0f}s', flush=True)
        if it % 2000 == 0 or it == iters: torch.save({'GAB': GAB.state_dict(), 'GBA': GBA.state_dict(), 'it': it}, _repo_path(H, f'cycgan{TAG}.pt'))
    print('DONE', flush=True)
elif mode == 'apply':
    K = int(sys.argv[2]); ck = torch.load(_repo_path(H, f'cycgan{TAG}.pt'), map_location='cpu'); GAB = UNet(2).to(dev); GAB.load_state_dict(ck['GAB']); GAB.eval()
    out = np.zeros((K, len(A_ids), 256, 256), np.uint8); gen = torch.Generator(device='cpu').manual_seed(7)
    with torch.no_grad():
        for n, i in enumerate(A_ids):
            x = t(IM[i])[None].to(dev)
            for k in range(K):
                z = torch.randn(x.shape, generator=gen).to(dev); y = GAB(torch.cat([x, z], 1))[0, 0].cpu().numpy() * 64 + 128; out[k, n] = np.clip(np.round(y), 0, 255).astype(np.uint8)
    np.savez_compressed(_repo_path(H, f'trans{TAG}-k{K}.npz'), ids=np.array(A_ids), images=out, it=ck['it'])
    nz = [noise_gray(out[0, n]) for n in range(len(A_ids))]; print('translated', len(A_ids), 'x', K, ' noise_gray q10/50/90', np.quantile(nz, [.1, .5, .9]).round(2),
                                                                     ' real B', np.quantile([NZ[i] for i in B_ids], [.1, .5, .9]).round(2))
