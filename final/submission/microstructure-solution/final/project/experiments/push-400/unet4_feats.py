import sys, csv, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
HERE = Path(__file__).resolve().parent; P = HERE.parents[1]
tag, split, names = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
src = open(HERE / 'phase_unet4.py').read()
import torch.nn as nn, torch.nn.functional as F
exec(src[src.index('def blk'):src.index("if __name__ == '__main__':")])
dev = torch.device('mps')
LA_MU, LA_SD = float(__import__("os").environ.get("LA_MU", 5.636784553527832)), float(__import__("os").environ.get("LA_SD", 0.9691801071166992))
nets = []
for n in names:
    w = 48 if 'w48' in n else 32; net = UNet(w).to(dev); net.load_state_dict(torch.load(HERE / f'unet4-{n}.pt', map_location=dev)); net.eval(); nets.append(net)
def norm(f):
    lo, med, hi = np.percentile(f, [5, 50, 95]); return ((f - med) / max(hi - lo, 1.0)).astype(np.float32)
def inv(p, k):
    """undo view k on maps p [C,H,W]; channels 4..6 = (logA, c2, s2): undo the doubled-angle transform too."""
    p = p.copy()
    if k & 4: p = p[:, :, ::-1]; p[6] = -p[6]
    if k & 2: p = p[:, ::-1]; p[6] = -p[6]
    if k & 1: p = p.transpose(0, 2, 1); p[5] = -p[5]
    return p
def predict(a):
    x = norm(a); views = []
    for k in range(8):
        v = x
        if k & 1: v = v.T
        if k & 2: v = v[::-1]
        if k & 4: v = v[:, ::-1]
        views.append(np.ascontiguousarray(v))
    xb = torch.tensor(np.stack(views))[:, None].to(dev); acc = np.zeros((8, 256, 256))
    with torch.no_grad():
        for net in nets:
            o = net(xb); pr = o[:, :4].softmax(1); rg = o[:, 4:].clone(); rg[:, 0] = rg[:, 0] * LA_SD + LA_MU; rg[:, 1:3] = rg[:, 1:3] * 0.3; rg[:, 3] = rg[:, 3] * 0.15 + 0.15
            out = torch.cat([pr, rg], 1).cpu().numpy()
            for k in range(8): acc += inv(out[k], k)
    return acc / (8 * len(nets))
KEYS = ['z_f20', 'z_f25', 'z_f30', 'z_f40', 'z_f50', 'z_mean', 'z_lin', 'm_ph', 'm_ph_hard', 'm_ph_f32', 'm_pore', 'm_bnd', 'm_la_mean', 'm_la_std', 'm_la_f16', 'm_la_f32', 'm_la_rng32', 'm_la_tile8', 'm_la_plane_r2',
        'm_el_glob', 'm_el_pix', 'm_el_loc16', 'm_el_loc32', 'm_el_coh', 'm_la_ph_minus_mx', 'm_inv_sqrt_d']
ids = [r['ID'] for r in csv.DictReader(open(P / 'data' / ('train.csv' if split == 'train' else 'sample_submission.csv')))]
yy, xx = np.mgrid[0:256, 0:256]
X = []; maps = np.zeros((len(ids), 8, 128, 128), np.float16)
for n, i in enumerate(ids):
    a = np.array(Image.open(P / f'data/{split}/{i}.png')).astype(np.float32); p = predict(a); maps[n] = p[:, ::2, ::2]
    g = p[0] + p[1]; o = {}
    zm = p[7]
    for t in (0.2, 0.25, 0.3, 0.4, 0.5): o[f'z_f{int(round(t * 100))}'] = (g / (1 + np.exp(-(zm - t) / 0.02))).sum() / g.sum()
    o['z_mean'] = (g * zm).sum() / g.sum(); o['z_lin'] = (g * np.clip((zm - 0.15) / 0.2, 0, 1)).sum() / g.sum()
    o['m_ph'] = p[1].sum() / max(g.sum(), 1); am = p[:4].argmax(0); o['m_ph_hard'] = (am == 1).sum() / max(((am == 0) | (am == 1)).sum(), 1)
    o['m_ph_f32'] = (ndi.gaussian_filter(p[1], 32) / np.maximum(ndi.gaussian_filter(g, 32), 1e-3)).std()
    o['m_pore'] = p[2].mean(); o['m_bnd'] = p[3].mean()
    wgt = g / g.sum(); la = p[4]
    o['m_la_mean'] = (wgt * la).sum(); o['m_la_std'] = np.sqrt((wgt * (la - o['m_la_mean']) ** 2).sum())
    o['m_inv_sqrt_d'] = (wgt * np.exp(-0.25 * la)).sum()
    for s in (16, 32):
        f = ndi.gaussian_filter(la * g, s) / np.maximum(ndi.gaussian_filter(g, s), 1e-3); o[f'm_la_f{s}'] = f.std()
        if s == 32:
            o['m_la_rng32'] = np.percentile(f, 95) - np.percentile(f, 5)
            D = np.c_[np.ones(65536), yy.ravel(), xx.ravel()]; c, *_ = np.linalg.lstsq(D, f.ravel(), rcond=None); r = f.ravel() - D @ c; o['m_la_plane_r2'] = 1 - r.var() / max(f.var(), 1e-9)
    t = (la * g).reshape(8, 32, 8, 32).sum((1, 3)) / np.maximum(g.reshape(8, 32, 8, 32).sum((1, 3)), 1e-3); o['m_la_tile8'] = t.std()
    c2, s2 = p[5] * g, p[6] * g
    o['m_el_glob'] = np.hypot(c2.sum(), s2.sum()) / g.sum(); o['m_el_pix'] = (wgt * np.hypot(p[5], p[6])).sum()
    for s in (16, 32):
        cc = ndi.gaussian_filter(c2, s) / np.maximum(ndi.gaussian_filter(g, s), 1e-3); ss = ndi.gaussian_filter(s2, s) / np.maximum(ndi.gaussian_filter(g, s), 1e-3)
        o[f'm_el_loc{s}'] = (wgt * np.hypot(cc, ss)).sum()
    o['m_el_coh'] = o['m_el_glob'] / max(o['m_el_loc16'], 1e-6)
    o['m_la_ph_minus_mx'] = (p[1] * la).sum() / max(p[1].sum(), 1e-3) - (p[0] * la).sum() / max(p[0].sum(), 1e-3)
    X.append([float(o[k]) for k in KEYS])
    if (n + 1) % 250 == 0: print(n + 1, flush=True)
np.savez(__import__("pathlib").Path(__import__("os").environ.get("OUTDIR", str(HERE))) / f'unet4-{tag}-{split}.npz', ids=np.array(ids), X=np.array(X), keys=np.array(KEYS))
if not __import__("os").environ.get("NOMAPS"): np.savez_compressed(HERE / f'unet4maps-{tag}-{split}.npz', ids=np.array(ids), maps=maps)
print('done', np.array(X).shape)
