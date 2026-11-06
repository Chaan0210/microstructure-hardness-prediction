"""Does flat-field (illumination) correction of the INPUT improve U-Net phase readouts on real noisy images?
Runs unet4-z1 (D4 TTA) on train images after correction MODE and saves readouts; label-free.
usage: flat_probe.py MODE   MODE: none | q2 (robust quadratic background on bright pixels) | q3 | g48 (gaussian sigma 48 background)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
HERE = Path(__file__).resolve().parent; P = _RepoPath(__file__).resolve().parents[3]; U = _repo_path(P, 'experiments/push-400')
mode = sys.argv[1]; net_name = sys.argv[2] if len(sys.argv) > 2 else 'z1'
src = open(_repo_path(U, 'phase_unet4.py')).read()
import torch.nn as nn, torch.nn.functional as F
exec(src[src.index('def blk'):src.index("if __name__ == '__main__':")])
dev = torch.device('mps')
net = UNet(32).to(dev); net.load_state_dict(torch.load(_repo_path(U, f'unet4-{net_name}.pt'), map_location=dev)); net.eval()
LA_MU, LA_SD = 5.636784553527832, 0.9691801071166992
def norm(f):
    lo, med, hi = np.percentile(f, [5, 50, 95]); return ((f - med) / max(hi - lo, 1.0)).astype(np.float32)
def inv(p, k):
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
        o = net(xb); pr = o[:, :4].softmax(1); rg = o[:, 4:].clone(); rg[:, 0] = rg[:, 0] * LA_SD + LA_MU; rg[:, 1:3] = rg[:, 1:3] * 0.3; rg[:, 3] = rg[:, 3] * 0.15 + 0.15
        out = torch.cat([pr, rg], 1).cpu().numpy()
        for k in range(8): acc += inv(out[k], k)
    return acc / 8
yy, xx = np.mgrid[0:256, 0:256] / 255.0 * 2 - 1
def background(a, deg):
    s = ndi.gaussian_filter(a, 3).ravel()
    terms = [np.ones(65536), xx.ravel(), yy.ravel()]
    if deg >= 2: terms += [xx.ravel() ** 2, yy.ravel() ** 2, (xx * yy).ravel()]
    if deg >= 3: terms += [xx.ravel() ** 3, yy.ravel() ** 3, (xx ** 2 * yy).ravel(), (xx * yy ** 2).ravel()]
    A = np.stack(terms, 1); w = np.ones(65536, bool)
    for it in range(4):
        c, *_ = np.linalg.lstsq(A[w], s[w], rcond=None); r = s - A @ c
        w = r > np.percentile(r, 40)          # keep bright (matrix) pixels: phase, pores, boundaries excluded
    b = (A @ c).reshape(256, 256); return b - b.mean()
def correct(a):
    if mode == 'none': return a
    if mode in ('q2', 'q3'): return a - background(a, int(mode[1]))
    if mode == 'g48':
        b = ndi.gaussian_filter(a, 48); return a - (b - b.mean())
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]
R = {k: [] for k in ['soft', 'argmax', 'z_f25', 'z_f50', 'z_f30', 'z_mean', 'bg_sd']}
for n, i in enumerate(ids):
    a = np.array(Image.open(_repo_path(P, f'data/train/{i}.png'))).astype(np.float32); c = correct(a)
    R['bg_sd'].append(float(np.std(a - c)))
    p = predict(c); g = p[0] + p[1]; zm = p[7]
    R['soft'].append(p[1].sum() / g.sum()); am = p[:4].argmax(0); R['argmax'].append((am == 1).sum() / max(((am == 0) | (am == 1)).sum(), 1))
    for t in (0.25, 0.3, 0.5): R[f'z_f{int(round(t * 100))}'].append((g / (1 + np.exp(-(zm - t) / 0.02))).sum() / g.sum())
    R['z_mean'].append((g * zm).sum() / g.sum())
np.savez(_repo_io(f'flat-{mode}-{net_name}.npz'), ids=np.array(ids), **{k: np.array(v) for k, v in R.items()})
print('done', mode)
