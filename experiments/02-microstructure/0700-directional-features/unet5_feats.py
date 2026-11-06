"""D4-TTA inference of unet5 (phase_unet5.py: 8 unet4 channels + 2 direct exceedance heads) -> zp_f25, zp_f50 (direct, grain-prob weighted)
and zr_f25, zr_f50 (threshold of the same net's regressed z, as in unet4_feats). usage: unet5_feats.py NAME SPLIT -> unet5-NAME-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
import torch.nn as nn, torch.nn.functional as F
name, split = sys.argv[1], sys.argv[2]
src = open(_repo_io('../push-400/phase_unet4.py')).read(); exec(src[src.index('def blk'):src.index("if __name__ == '__main__':")])
dev = torch.device('mps'); net = UNet(32, nout=10).to(dev); net.load_state_dict(torch.load(_repo_io(f'unet5-{name}.pt'), map_location=dev)); net.eval()
LA_MU, LA_SD = 5.636784553527832, 0.9691801071166992
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
    xb = torch.tensor(np.stack(views))[:, None].to(dev); acc = np.zeros((10, 256, 256))
    with torch.no_grad():
        o = net(xb); pr = o[:, :4].softmax(1); rg = o[:, 4:8].clone(); rg[:, 0] = rg[:, 0] * LA_SD + LA_MU; rg[:, 1:3] = rg[:, 1:3] * 0.3; rg[:, 3] = rg[:, 3] * 0.15 + 0.15
        out = torch.cat([pr, rg, torch.sigmoid(o[:, 8:10])], 1).cpu().numpy()
        for k in range(8): acc += inv(out[k], k)
    return acc / 8
ids = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/' + ('train.csv' if split == 'train' else 'sample_submission.csv'))))]
X = []
for n, i in enumerate(ids):
    p = predict(np.array(Image.open(_repo_io(f'../../data/{split}/{i}.png'))).astype(np.float32)); g = p[0] + p[1]
    zr = [(g / (1 + np.exp(-(p[7] - t) / 0.02))).sum() / g.sum() for t in (0.25, 0.5)]
    zp = [(g * p[8]).sum() / g.sum(), (g * p[9]).sum() / g.sum()]
    X.append(zp + zr)
    if (n + 1) % 250 == 0: print(n + 1, flush=True)
np.savez(_repo_io(f'unet5-{name}-{split}.npz'), ids=np.array(ids), X=np.array(X), keys=np.array(['zp_f25', 'zp_f50', 'zr_f25', 'zr_f50']))
print('done', split, np.array(X).mean(0).round(4))
