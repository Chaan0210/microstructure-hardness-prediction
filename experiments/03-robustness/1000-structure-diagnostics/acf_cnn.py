"""Branch 2, tiny conv model on the unsummarised correlation maps (acf-train.npz, 4 x 33 x 33 + 4 amplitudes), dev protocol.
prediction = Ridge skip (RidgeCV on X98, fitted on the outer-training rows) + zero-initialised correction. Correction inputs:
  struct  12 core structure columns only (control)
  conv    conv(4->8,3) SiLU maxpool2 conv(8->16,3) SiLU global mean+max -> concat 12 structure columns + 4 amplitudes
Inner 20% split selects the epoch count (max 200, patience 25), then refit on all outer-training rows; AdamW 1e-3, wd 1e-2, batch 32, MSE on
standardised targets. usage: acf_cnn.py VARIANT SEED[,SEED...] -> acfcnn-<VARIANT>-s<seed>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, torch, torch.nn as nn, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, _repo_io('../push-900'))
from blend_dev import load
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
torch.set_num_threads(2)
VAR = sys.argv[1]; SEEDS = [int(s) for s in sys.argv[2].split(',')]
B = load(9); ids, y, FO = B['ids'], B['y'], B['fold']
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; fp = {str(i): k for k, i in enumerate(F['ids'])}; X98 = F['X'][[fp[i] for i in ids]].astype(float)
CORE = ['u_ph', 'm_ph_hard', 'aw_inv_sqrt_d', 'grain_inv_sqrt_d', 'het4_u', 'pore_frac', 'ar_wmean', 'orient_coherence', 'ang_c4', 'n_obj', 'sh_cor15', 'noise']
S = X98[:, [fk.index(c) for c in CORE]]
Q = np.load(_repo_io('acf-train.npz')); qp = {str(i): k for k, i in enumerate(Q['ids'])}; A = Q['A'][[qp[i] for i in ids]].astype(np.float32); AMP = Q['amp'][[qp[i] for i in ids]].astype(float)
class Net(nn.Module):
    def __init__(s, ng):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.SiLU()) if VAR == 'conv' else None
        nin = (32 if VAR == 'conv' else 0) + ng
        s.head = nn.Sequential(nn.Linear(nin, 32), nn.SiLU(), nn.Dropout(0.1), nn.Linear(32, 1)); nn.init.zeros_(s.head[-1].weight); nn.init.zeros_(s.head[-1].bias)
    def forward(s, a, g):
        if s.conv is not None: h = s.conv(a); g = torch.cat([h.mean((2, 3)), h.amax((2, 3)), g], 1)
        return s.head(g).squeeze(-1)
def train(Atr, Gtr, rtr, Ava, Gva, epochs, seed, val=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = Net(Gtr.shape[1]); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    At, Gt, rt = map(torch.from_numpy, (Atr, Gtr.astype(np.float32), rtr.astype(np.float32))); best, be, stale, hist = None, 0, 0, []
    def pred(Ax, Gx): net.eval(); return net(torch.from_numpy(Ax), torch.from_numpy(Gx.astype(np.float32))).detach().numpy()
    if val is not None: best = np.mean((pred(*val[:2]) - val[2]) ** 2)
    for e in range(1, epochs + 1):
        net.train(); o = rng.permutation(len(rt))
        for k in range(0, len(o), 32):
            b = o[k:k + 32]; loss = ((net(At[b], Gt[b]) - rt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if val is not None:
            v = np.mean((pred(*val[:2]) - val[2]) ** 2)
            if v < best - 1e-5: best, be, stale = v, e, 0
            else: stale += 1
            if stale >= 25: break
    return (be if val is not None else None), pred(Ava, Gva)
out = {}
for seed in SEEDS:
    P = np.zeros((3, 400))
    for k in range(3):
        for f in range(5):
            tr, va = FO[k] != f, FO[k] == f; tri = np.where(tr)[0]
            rg = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(X98[tr], y[tr]); skip = rg.predict(X98)
            gsc = make_pipeline(SimpleImputer(strategy='median'), StandardScaler()).fit(np.c_[S, AMP][tr]); G = np.clip(gsc.transform(np.c_[S, AMP]), -5, 5)
            ym, ys = y[tr].mean(), y[tr].std(); r = (y - skip) / ys
            rs = np.random.default_rng(seed + 100 * k + f).permutation(tri); iv, it = rs[:len(rs) // 5], rs[len(rs) // 5:]
            rgi = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(X98[it], y[it]); ri = (y - rgi.predict(X98)) / ys
            ep, _ = train(A[it], G[it], ri[it], A[iv], G[iv], 200, seed, val=(A[iv], G[iv], ri[iv]))
            _, corr = train(A[tr], G[tr], r[tr], A[va], G[va], max(ep, 0), seed) if ep > 0 else (0, np.zeros(va.sum()))
            P[k, va] = skip[va] + corr * ys
    np.savez(_repo_io(f'acfcnn-{VAR}-s{seed}.npz'), oof=P); out[seed] = P; print(VAR, seed, 'done', flush=True)
