"""Image descriptors from a fine-tuned Cellpose-SAM: neck output (256 ch on the 32x32 token grid) pooled per image
(mean, std, 10/50/90% quantiles per channel -> 1280 dims), averaged over D4 views (id, T, UD, R90). Label-free; test only passed forward.
usage: embed_ft.py CKPT SPLIT  -> emb-<ckpt>-<split>.npz (ids, E)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from PIL import Image
import csv
from cellpose import models, transforms
ckpt, split = sys.argv[1], sys.argv[2]
P = Path(_repo_io('../..'))
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv' if split == 'train' else 'data/sample_submission.csv')))]
M = models.CellposeModel(gpu=True, pretrained_model=ckpt); net = M.net; net.eval(); dev = M.device
V = [lambda a: a, lambda a: a.T, lambda a: a[::-1], lambda a: np.rot90(a)]
def norm(u8):
    return transforms.normalize_img(u8.astype(np.float32)[..., None], normalize=True, axis=-1)[..., 0]
@torch.no_grad()
def neck(x):
    x = torch.nn.functional.conv2d(x, net.encoder.patch_embed.proj.weight[:, :1], bias=net.encoder.patch_embed.proj.bias, stride=net.ps)
    x = x.permute(0, 2, 3, 1) + net.encoder.pos_embed
    for blk in net.encoder.blocks: x = blk(x)
    return net.encoder.neck(x.permute(0, 3, 1, 2))     # [B, 256, 32, 32]
out = []; t0 = time.time()
for n, i in enumerate(ids):
    im = np.array(Image.open(_repo_path(P, f'data/{split}/{i}.png')))
    X = torch.from_numpy(np.stack([norm(np.ascontiguousarray(f(im))) for f in V])[:, None]).to(dev, dtype=net.dtype)
    z = neck(X).float().flatten(2)                     # [4, 256, 1024]
    q = torch.quantile(z, torch.tensor([0.1, 0.5, 0.9], device=z.device), dim=2)   # [3, 4, 256]
    feat = torch.cat([z.mean(2), z.std(2), q[0], q[1], q[2]], 1).mean(0)            # average over views
    out.append(feat.cpu().numpy())
    if (n + 1) % 100 == 0: print(json.dumps(dict(done=n + 1, n=len(ids), sec=round(time.time() - t0))), flush=True)
np.savez(_repo_io(f'emb-{Path(ckpt).stem}-{split}.npz'), ids=np.array(ids), E=np.array(out, np.float32)); print('DONE', flush=True)
