# infomatched-copies-20261016: copy of push-1200 script, inputs/outputs redirected to this experiment (W/synth, W/feat); models and
# reference files are read from their original locations unchanged.
"""U-Net features for synthetic degraded copies (push-400/synth/<SET>/*.png): multi-task nets (unet2) + z net (unet4), D4 TTA, maps saved.
usage: synth_unet.py SET NET  (NET: m1|m2|h1|m3 -> unet2 ; u4z1 -> unet4)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, numpy as np, torch, glob
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
W = Path(__file__).resolve().parents[1]; H = _repo_path(W, 'feat'); U = _repo_path(W.parent, 'push-400')
st, net = sys.argv[1], sys.argv[2]
is4 = net.startswith('u4'); nm = net[2:] if is4 else net
src = open(_repo_path(U, 'phase_unet4.py' if is4 else 'phase_unet2.py')).read(); g = {}
exec('import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\n' + src[src.index('def blk'):src.index("if __name__ == '__main__':")], g)
dev = torch.device('mps'); model = g['UNet'](32).to(dev); model.load_state_dict(torch.load(_repo_path(U, f'unet4-{nm}.pt' if is4 else f'unet2-{nm}.pt'), map_location=dev)); model.eval()
LA = {'m1': (5.636784553527832, 0.9691801071166992), 'h1': (5.636784553527832, 0.9691801071166992), 'z1': (5.636784553527832, 0.9691801071166992),
      'm2': (5.6343302726745605, 0.9692867994308472), 'm3': (5.6343302726745605, 0.9692867994308472)}[nm]
C = 8 if is4 else 7
def norm(f): lo, md, hi = np.percentile(f, [5, 50, 95]); return ((f - md) / max(hi - lo, 1.0)).astype(np.float32)
def inv(p, k):
    p = p.copy()
    if k & 4: p = p[:, :, ::-1]; p[6] = -p[6]
    if k & 2: p = p[:, ::-1]; p[6] = -p[6]
    if k & 1: p = p.transpose(0, 2, 1); p[5] = -p[5]
    return p
files = sorted(glob.glob(str(_repo_path(W, f'synth/{st}/*.png')))); ids = [Path(f).stem for f in files]
maps = np.zeros((len(ids), C, 128, 128), np.float16)

yy, xx = np.mgrid[0:256, 0:256]
def feats(p):
    g = p[0] + p[1]; o = {}
    if is4:
        zm = p[7]
        for t in (0.25, 0.5): o[f'z_f{int(round(t * 100))}'] = (g / (1 + np.exp(-(zm - t) / 0.02))).sum() / g.sum()
    am = p[:4].argmax(0); o['m_ph_hard'] = (am == 1).sum() / max(((am == 0) | (am == 1)).sum(), 1)
    wgt = g / g.sum(); la = p[4]
    o['m_la_mean'] = (wgt * la).sum(); o['m_la_std'] = np.sqrt((wgt * (la - o['m_la_mean']) ** 2).sum())
    for s in (16, 32):
        f = ndi.gaussian_filter(la * g, s) / np.maximum(ndi.gaussian_filter(g, s), 1e-3); o[f'm_la_f{s}'] = f.std()
    t = (la * g).reshape(8, 32, 8, 32).sum((1, 3)) / np.maximum(g.reshape(8, 32, 8, 32).sum((1, 3)), 1e-3); o['m_la_tile8'] = t.std()
    c2, s2 = p[5] * g, p[6] * g
    o['m_el_glob'] = np.hypot(c2.sum(), s2.sum()) / g.sum(); o['m_el_pix'] = (wgt * np.hypot(p[5], p[6])).sum()
    cc = ndi.gaussian_filter(c2, 16) / np.maximum(ndi.gaussian_filter(g, 16), 1e-3); ss = ndi.gaussian_filter(s2, 16) / np.maximum(ndi.gaussian_filter(g, 16), 1e-3)
    o['m_el_loc16'] = (wgt * np.hypot(cc, ss)).sum(); o['m_el_coh'] = o['m_el_glob'] / max(o['m_el_loc16'], 1e-6)
    return o
FK = None; FX = []
for n, fn in enumerate(files):
    x = norm(np.array(Image.open(_repo_io(fn))).astype(np.float32)); vs = []
    for k in range(8):
        v = x
        if k & 1: v = v.T
        if k & 2: v = v[::-1]
        if k & 4: v = v[:, ::-1]
        vs.append(np.ascontiguousarray(v))
    with torch.no_grad():
        o = model(torch.tensor(np.stack(vs))[:, None].to(dev)); pr = o[:, :4].softmax(1); rg = o[:, 4:].clone(); rg[:, 0] = rg[:, 0] * LA[1] + LA[0]; rg[:, 1:3] = rg[:, 1:3] * 0.3
        if is4: rg[:, 3] = rg[:, 3] * 0.15 + 0.15
        out = torch.cat([pr, rg], 1).cpu().numpy()
    acc = np.zeros((C, 256, 256))
    for k in range(8): acc += inv(out[k], k)
    p = acc / 8; maps[n] = p[:, ::2, ::2]; o = feats(p); FK = list(o); FX.append([float(o[k]) for k in FK])
np.savez_compressed(_repo_path(H, f'synthmaps-{st}-{net}.npz'), ids=np.array(ids), maps=maps); np.savez(_repo_path(H, f'synthfeat-{st}-{net}.npz'), ids=np.array(ids), X=np.array(FX), keys=np.array(FK)); print('done', st, net, len(ids))
