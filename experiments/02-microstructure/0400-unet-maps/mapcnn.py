"""Small CNN regressor on U-Net measurement maps (7ch @128: P(matrix), P(phase), P(pore), P(boundary), logA, c2, s2) + noise scalar.
Harness folds (500 rows, reps given). Saves OOF. usage: mapcnn.py MAPS_NPZ REPS [epochs=60] [seeds=1] [tag]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, _repo_io('../push-300'))
import harness
mp, reps = sys.argv[1], [int(r) for r in sys.argv[2].split(',')]; epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 60
nseeds = int(sys.argv[4]) if len(sys.argv) > 4 else 1; tag = sys.argv[5] if len(sys.argv) > 5 else 'mapcnn'
D = np.load(_repo_io(mp), allow_pickle=True); ids = [str(i) for i in D['ids']]; M = D['maps'].astype(np.float32)
y = harness.labels(ids); nz = np.load(_repo_io('../push-300/noise_train.npy')); dev = torch.device('mps')
if M.shape[1] == 7: M[:, 4] = (M[:, 4] - 5.6) / 1.0; M[:, 5:] = M[:, 5:] / 0.3
ym, ys = y.mean(), y.std()
def aug(x, k):   # D4 on [B,C,H,W]; fix doubled-angle channels (5=c2, 6=s2) when present
    if k & 1: x = x.transpose(2, 3); x = x.clone(); 
    if k & 1 and x.shape[1] == 7: x[:, 5] = -x[:, 5]
    if k & 2: x = x.flip(2); 
    if k & 2 and x.shape[1] == 7: x[:, 6] = -x[:, 6]
    if k & 4: x = x.flip(3)
    if k & 4 and x.shape[1] == 7: x[:, 6] = -x[:, 6]
    return x
class Net(nn.Module):
    def __init__(s, cin):
        super().__init__()
        def cb(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU())
        s.f = nn.Sequential(cb(cin, 32), cb(32, 32), nn.MaxPool2d(2), cb(32, 64), cb(64, 64), nn.MaxPool2d(2), cb(64, 96), cb(96, 96), nn.MaxPool2d(2), cb(96, 128))
        s.h = nn.Sequential(nn.Linear(128 * 2 + 1, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(s, x, z):
        f = s.f(x); g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), z[:, None]], 1); return s.h(g)[:, 0]
Z = (nz - nz.mean()) / nz.std()
O = np.zeros((len(reps), len(y)))
for ri, r in enumerate(reps):
    f = harness.folds(len(y), r)
    for j in range(5):
        tr, va = np.where(f != j)[0], np.where(f == j)[0]; P = []
        for sd in range(nseeds):
            torch.manual_seed(100 * sd + j); net = Net(M.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
            steps = epochs * int(np.ceil(len(tr) / 16)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
            Xt = torch.tensor(M[tr]).to(dev); Yt = torch.tensor((y[tr] - ym) / ys, dtype=torch.float32).to(dev); Zt = torch.tensor(Z[tr], dtype=torch.float32).to(dev)
            rng = np.random.RandomState(sd * 7 + j)
            for ep in range(epochs):
                net.train(); perm = rng.permutation(len(tr))
                for b in range(0, len(tr), 16):
                    idx = perm[b:b + 16]; xb = aug(Xt[idx], rng.randint(8))
                    loss = F.mse_loss(net(xb, Zt[idx]), Yt[idx]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            net.eval(); Xv = torch.tensor(M[va]).to(dev); Zv = torch.tensor(Z[va], dtype=torch.float32).to(dev)
            with torch.no_grad(): pv = np.mean([net(aug(Xv, k), Zv).cpu().numpy() for k in range(8)], 0)
            P.append(pv * ys + ym)
        O[ri, va] = np.mean(P, 0)
        print(f'rep {r} fold {j} rmse {np.sqrt(np.mean((O[ri, va] - y[va]) ** 2)):.3f}', flush=True)
np.save(_repo_io(f'oof_{tag}.npy'), O); harness.report(tag, O, y, nz)
B = np.stack([np.load(_repo_io(f'oof_v2_feat-u1ph-train_s{s}.npy')) for s in range(3)]).mean(0)[[0, 1, 2].index(reps[0]) if len(reps) == 1 else slice(None)]
B = B if B.ndim == 2 else B[None]
harness.report('tabpfn u1ph', B, y, nz)
for a in (0.2, 0.3, 0.4): harness.report(f'blend tabpfn*{1-a:.1f}+cnn*{a:.1f}', (1 - a) * B + a * O, y, nz)
print('resid corr', np.mean([np.corrcoef(O[i] - y, B[i] - y)[0, 1] for i in range(len(O))]))
