"""Label-free feature-restoration CNN ("distillation").
Teacher: Cellpose-SAM v3 features measured on cleaner train images (noise <= 60th pct).
Student: CNN that sees a randomly degraded (blur / noise / contrast / brightness) + D4-transformed copy and predicts the teacher's
clean features. Hardness labels are never used; test images are never used for training (only inference at the end).
Output: restored-feature predictions (8-view D4 mean) for all train and test images -> distill-<tag>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '.')
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--tag', default='a'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--threads', type=int, default=2); ap.add_argument('--holdout', type=float, default=0.1); ap.add_argument('--width', type=int, default=32)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
P = _RepoPath(__file__).resolve().parents[3]; G = _repo_path(P, 'experiments/grain-cellpose-220')
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
TARGETS = ['sp_frac_grains', 'sp_count_frac', 'f_dark_20', 'f_dark_35', 'f_dark_50', 'z_q75', 'sp_contiguity', 'sp_field_std_s32',
           'sizefield_range_s32', 'sizefield_std_s32', 'sizefield_std_s16', 'tile8_logA_std', 'tile_logA_std', 'grain_logA_std',
           'grain_eqd_q90_q10', 'coarse_area_frac', 'fine_area_frac', 'logA_skew', 'grain_eqd_mean', 'grain_eqd_med', 'aw_eqd_mean',
           'aw_eqd_q90', 'grain_density_j', 'ar_wmean', 'ar_q90', 'orient_coherence', 'boundary_coherence', 'solidity_wmean',
           'neighbors_std', 'px_pore_n', 'px_pore_frac', 'pore_frac', 'sp_to_matrix_size', 'sp_ar']
Ftr = np.load(_repo_path(G, 'features-v3-train.npz'), allow_pickle=True); keys = [str(k) for k in Ftr['keys']]; ids = [str(i) for i in Ftr['ids']]
T = Ftr['X'][:, [keys.index(k) for k in TARGETS]].astype(np.float64)
signed = np.array([k == 'logA_skew' for k in TARGETS])
T[:, ~signed] = np.log1p(np.clip(T[:, ~signed], 0, None))
nz = np.load(_repo_path(Path(__file__).resolve().parent, 'noise_train.npy'))
ok = np.isfinite(T).all(1) & (nz <= np.quantile(nz, 0.6))
src = np.where(ok)[0]; rng.shuffle(src)
n_ho = int(len(src) * a.holdout); ho, trs = src[:n_ho], src[n_ho:]
mu, sd = T[trs].mean(0), T[trs].std(0) + 1e-6; Tz = (T - mu) / sd
imgs_tr = np.stack([np.asarray(Image.open(_repo_path(P, f'data/train/{i}.png')), np.float32) for i in ids])

def degrade(f, r):
    if r.rand() < 0.15:
        g = f.copy(); sn = 0.0
    else:
        sb, sn, c = r.uniform(0.3, 2.0), r.uniform(4, 22), r.uniform(0.65, 1.0)
        g = ndi.gaussian_filter(f, sb); m = g.mean(); g = m + c * (g - m) + r.normal(0, sn, f.shape)
    g = g + r.uniform(-20, 20)
    return np.clip(np.round(g), 0, 255)

def d4(x, k):
    if k >= 4: x = x[:, ::-1]
    return np.rot90(x, k % 4)

def norm(x):
    x = x.astype(np.float32); return (x - x.mean()) / (x.std() + 1e-3)

class Block(nn.Module):
    def __init__(s, ci, co, stride):
        super().__init__()
        s.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False); s.n1 = nn.GroupNorm(8, co)
        s.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); s.n2 = nn.GroupNorm(8, co)
        s.sk = nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False), nn.GroupNorm(8, co)) if (ci != co or stride != 1) else nn.Identity()
    def forward(s, x):
        h = Fn.gelu(s.n1(s.c1(x))); h = s.n2(s.c2(h)); return Fn.gelu(h + s.sk(x))

class Net(nn.Module):
    def __init__(s, w, k):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(1, w, 5, 1, 2, bias=False), nn.GroupNorm(8, w), nn.GELU())
        s.body = nn.Sequential(Block(w, w, 2), Block(w, 2 * w, 2), Block(2 * w, 2 * w, 1), Block(2 * w, 4 * w, 2), Block(4 * w, 4 * w, 1),
                               Block(4 * w, 8 * w, 2), Block(8 * w, 8 * w, 1))
        s.head = nn.Sequential(nn.Linear(16 * w, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, k))
    def forward(s, x):
        h = s.body(s.stem(x)); return s.head(torch.cat([h.mean((2, 3)), h.std((2, 3))], 1))

net = Net(a.width, len(TARGETS)).to(dev)
steps_per_epoch = int(np.ceil(len(trs) * 2 / 32)); total = a.epochs * steps_per_epoch
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.02)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total, pct_start=0.1)
t0 = time.time(); step = 0
ho_fixed = [(i, degrade(imgs_tr[i], np.random.RandomState(1000 + i)), (1000 + i) % 8) for i in ho]

def evaluate():
    net.eval()
    with torch.no_grad():
        X = torch.from_numpy(np.stack([norm(d4(g, k).copy()) for _, g, k in ho_fixed])[:, None]).to(dev)
        p = net(X).cpu().numpy()
    net.train(); y = Tz[[i for i, _, _ in ho_fixed]]
    r2 = 1 - ((p - y) ** 2).mean(0) / y.var(0)
    return float(np.mean(r2)), r2

for ep in range(a.epochs):
    order = np.concatenate([rng.permutation(trs), rng.permutation(trs)])
    for b in range(0, len(order), 32):
        bi = order[b:b + 32]
        xb = np.stack([norm(d4(degrade(imgs_tr[i], rng), rng.randint(8)).copy()) for i in bi])[:, None]
        xb = torch.from_numpy(xb).to(dev); yb = torch.from_numpy(Tz[bi].astype(np.float32)).to(dev)
        loss = Fn.mse_loss(net(xb), yb); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sch.step(); step += 1
    if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
        m, _ = evaluate(); print(json.dumps(dict(ep=ep + 1, loss=round(float(loss), 4), ho_meanR2=round(m, 4), sec=round(time.time() - t0))), flush=True)

m, r2 = evaluate(); print('holdout R2 per target:', json.dumps({k: round(float(v), 3) for k, v in zip(TARGETS, r2)}), flush=True)
torch.save(net.state_dict(), _repo_path(Path(__file__).resolve().parent, f'distill-{a.tag}.pt'))

def predict(split):
    files = sorted((_repo_path(P, f'data/{split}')).glob('*.png')); out = np.zeros((len(files), len(TARGETS))); net.eval()
    with torch.no_grad():
        for s in range(0, len(files), 16):
            ims = [np.asarray(Image.open(_repo_io(f)), np.float32) for f in files[s:s + 16]]
            acc = 0
            for k in range(8):
                X = torch.from_numpy(np.stack([norm(d4(im, k).copy()) for im in ims])[:, None]).to(dev); acc = acc + net(X).cpu().numpy()
            out[s:s + 16] = acc / 8
    return [f.stem for f in files], out * sd + mu

tr_ids, ptr = predict('train'); te_ids, pte = predict('test')
np.savez(_repo_path(Path(__file__).resolve().parent, f'distill-{a.tag}.npz'), train_ids=np.array(tr_ids), train=ptr, test_ids=np.array(te_ids), test=pte,
         keys=np.array(['dst_' + k for k in TARGETS]), holdout=np.array(ho), ho_r2=r2)
print('DONE', round(time.time() - t0), flush=True)
