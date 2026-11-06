"""Diagnostic: angle features of original / fliplr / rot90 images through the same z1 U-Net with D4 TTA; full-res vs [::2,::2] maps."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, torch, csv
from pathlib import Path
from PIL import Image
import torch.nn as nn, torch.nn.functional as F
src = open(_repo_io('../push-400/phase_unet4.py')).read(); exec(src[src.index('def blk'):src.index("if __name__ == '__main__':")])
dev = torch.device('mps'); net = UNet(32).to(dev); net.load_state_dict(torch.load(_repo_io('../push-400/unet4-z1.pt'), map_location=dev)); net.eval()
def norm(f): lo, med, hi = np.percentile(f, [5, 50, 95]); return ((f - med) / max(hi - lo, 1.0)).astype(np.float32)
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
        o = net(xb); pr = o[:, :4].softmax(1); rg = o[:, 4:].clone(); rg[:, 1:3] = rg[:, 1:3] * 0.3
        out = torch.cat([pr, rg], 1).cpu().numpy()
        for k in range(8): acc += inv(out[k], k)
    return acc / 8
def ang(p):
    g = p[0] + p[1]; C = (g * p[5]).sum() / g.sum(); S = (g * p[6]).sum() / g.sum(); v = np.hypot(C, S) + 1e-12
    return np.array([(C * C - S * S) / v, 2 * C * S / v])
ids = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/train.csv')))]
A = np.load(_repo_io('angle-train.npz'))['X']; pick = np.argsort(-np.hypot(A[:, 0], A[:, 1]))[:12].tolist() + list(range(0, 500, 63))
rows = []
for n in pick:
    a = np.array(Image.open(_repo_io(f'../../data/train/{ids[n]}.png'))).astype(np.float32)
    res = {}
    for nm, b in [('orig', a), ('fliplr', a[:, ::-1]), ('rot90', np.rot90(a))]:
        p = predict(np.ascontiguousarray(b)); res[nm] = (ang(p), ang(p[:, ::2, ::2]))
    o_f, o_s = res['orig']; l_f, l_s = res['fliplr']; r_f, r_s = res['rot90']
    rows.append([o_f[0], o_f[1], l_f[0], l_f[1], r_f[0], r_f[1], o_s[0], o_s[1], l_s[0], l_s[1], r_s[0], r_s[1]])
R = np.array(rows)
print('n images', len(R))
print('FULL-RES  fliplr: |c4 - c4_orig| mean %.4f, |s4 + s4_orig| mean %.4f | rot90: |c4 diff| %.4f |s4 diff| %.4f   (feature sd ~0.4)' % (np.abs(R[:, 2] - R[:, 0]).mean(), np.abs(R[:, 3] + R[:, 1]).mean(), np.abs(R[:, 4] - R[:, 0]).mean(), np.abs(R[:, 5] - R[:, 1]).mean()))
print('DOWNSAMPLED fliplr: |c4 diff| %.4f |s4 + s4_orig| %.4f | rot90: |c4 diff| %.4f |s4 diff| %.4f' % (np.abs(R[:, 8] - R[:, 6]).mean(), np.abs(R[:, 9] + R[:, 7]).mean(), np.abs(R[:, 10] - R[:, 6]).mean(), np.abs(R[:, 11] - R[:, 7]).mean()))
print('full vs downsampled (orig): |c4| %.4f |s4| %.4f' % (np.abs(R[:, 0] - R[:, 6]).mean(), np.abs(R[:, 1] - R[:, 7]).mean()))
