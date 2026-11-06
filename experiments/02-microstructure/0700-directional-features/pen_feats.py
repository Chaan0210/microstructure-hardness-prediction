"""Frozen z1 U-Net (push-400/unet4-z1.pt): penultimate 32-channel decoder representation (input to the final 1x1 conv) vs the 8 semantic outputs.
Per image (D4 TTA, rotation-invariant pooling only): grain-prob-weighted mean and std of each channel over the image -> pen (64), sem (16). Train images only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, torch, csv
from PIL import Image
import torch.nn as nn, torch.nn.functional as F
src = open(_repo_io('../push-400/phase_unet4.py')).read(); exec(src[src.index('def blk'):src.index("if __name__ == '__main__':")])
dev = torch.device('mps'); net = UNet(32).to(dev); net.load_state_dict(torch.load(_repo_io('../push-400/unet4-z1.pt'), map_location=dev)); net.eval()
feat = {}
net.out.register_forward_hook(lambda mod, inp, out: feat.__setitem__('pen', inp[0]))
def norm(f): lo, med, hi = np.percentile(f, [5, 50, 95]); return ((f - med) / max(hi - lo, 1.0)).astype(np.float32)
ids = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/train.csv')))]
PEN = np.zeros((500, 64)); SEM = np.zeros((500, 16))
for n, i in enumerate(ids):
    x = norm(np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(np.float32)); views = []
    for k in range(8):
        v = x
        if k & 1: v = v.T
        if k & 2: v = v[::-1]
        if k & 4: v = v[:, ::-1]
        views.append(np.ascontiguousarray(v))
    with torch.no_grad():
        o = net(torch.tensor(np.stack(views))[:, None].to(dev)); pen = feat['pen']
        pr = o[:, :4].softmax(1); g = (pr[:, 0] + pr[:, 1])[:, None]; W = g.sum((2, 3))
        sem = torch.cat([pr, o[:, 4:]], 1)
        def ms(t):   # per-view weighted mean/std over pixels, then averaged over the 8 views (pooled stats are invariant to the view transform)
            mu = (t * g).sum((2, 3)) / W; sd = torch.sqrt(((t - mu[:, :, None, None]) ** 2 * g).sum((2, 3)) / W); return torch.cat([mu, sd], 1).mean(0).cpu().numpy()
        PEN[n] = ms(pen); SEM[n] = ms(sem)
    if (n + 1) % 100 == 0: print(n + 1, flush=True)
np.savez(_repo_io('penfeat-train.npz'), ids=np.array(ids), PEN=PEN, SEM=SEM)
print('done', PEN.shape, SEM.shape)
