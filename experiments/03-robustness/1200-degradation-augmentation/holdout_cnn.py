"""Holdout-100 CNN arms: TinyCNN128 ctrl/dta (cnn_dta.py recipe) trained on the 400 Study900 dev rows (+ copies of dev sources, noise >= 12, d1to7),
predicting the 100 holdout rows. usage: holdout_cnn.py SEEDS -> holdout-cnn-<arm>-s<seed>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
P = Path(_repo_io('../..')); sys.path.insert(0, str(_repo_path(P, 'studies'))); from cnn_controls import TinyCNN; sys.path.pop(0)
F = torch.nn.functional; dev = 'mps'; torch.set_num_threads(2)
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
dset = set(str(i) for i in np.load(_repo_path(P, 'experiments/probabilistic-grains-900/base.npz'), allow_pickle=True)['ids'])
tr_ids = sorted(dset); ho_ids = sorted(set(lab) - dset)
img = lambda f: np.asarray(Image.open(_repo_io(f)).resize((128, 128), Image.Resampling.BOX), dtype=np.float32) / 255.
X = np.stack([img(_repo_path(P, 'data/train', f'{i}.png')) for i in tr_ids])[:, None]; y = np.array([lab[i] for i in tr_ids]); XH = np.stack([img(_repo_path(P, 'data/train', f'{i}.png')) for i in ho_ids])[:, None]
SX, SID = [], []
for s in 'deg,deg2,deg3,deg4,deg5,deg6,deg7'.split(','):
    S = np.load(_repo_io(f'synthrows-{s}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); k = [str(x) for x in S['keys']]; nz = S['X'][:, k.index('noise')].astype(float) * 255
    for i, n in zip(S['ids'], nz):
        if n >= 12 and str(i) in dset: SX.append(img(_repo_path(P, f'experiments/push-400/synth/train_{s}/{i}.png'))); SID.append(str(i))
SX = np.stack(SX)[:, None]; SY = np.array([lab[i] for i in SID]); print('train', len(tr_ids), 'holdout', len(ho_ids), 'copies', len(SID), flush=True)
def d4(xb, rng):
    xb = np.rot90(xb, int(rng.integers(4)), axes=(2, 3)); return np.ascontiguousarray(xb[:, :, :, ::-1] if rng.random() < .5 else xb)
@torch.no_grad()
def predict(m, x, nm):
    m.eval(); return np.concatenate([m(torch.from_numpy(np.ascontiguousarray((x[j:j + 64] - nm[0]) / nm[1])).to(dev)).cpu().numpy() for j in range(0, len(x), 64)]) * nm[3] + nm[2]
nm = (float(X.mean()), float(X.std()), float(y.mean()), float(y.std())); XN = (X - nm[0]) / nm[1]; yN = (y - nm[2]) / nm[3]; XS = (SX - nm[0]) / nm[1]; yS = (SY - nm[2]) / nm[3]
for seed in [int(s) for s in sys.argv[1].split(',')]:
    for arm in ('ctrl', 'dta'):
        out = Path(f'holdout-cnn-{arm}-s{seed}.npz')
        if out.exists(): continue
        t0 = time.time(); torch.manual_seed(1000 * seed + 55); m = TinyCNN().to(dev); o = torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=80, eta_min=.0001); rng = np.random.default_rng(1000 * seed + 55); rs = np.random.default_rng(7 + 1000 * seed + 55); avg = None
        for ep in range(80):
            perm = rng.permutation(len(y))
            for f0 in range(0, len(y), 32):
                ix = perm[f0:f0 + 32]; xb = d4(XN[ix], rng); yb = yN[ix]
                if arm == 'dta':
                    js = rs.integers(len(SID), size=32); xb = np.concatenate([xb, d4(XS[js], rs)]); yb = np.r_[yb, yS[js]]
                m.train(); o.zero_grad(set_to_none=True); loss = F.mse_loss(m(torch.from_numpy(xb).to(dev)), torch.tensor(yb, dtype=torch.float32, device=dev))
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.); o.step()
            sch.step()
            if ep >= 75:
                st = {n: v.detach().cpu().clone() for n, v in m.state_dict().items()}
                avg = {n: v / 5 for n, v in st.items()} if avg is None else {n: avg[n] + st[n] / 5 for n in st}
        m.load_state_dict(avg); ph = predict(m, XH, nm); np.savez(_repo_io(out), pred=ph, ids=np.array(ho_ids)); print(f's{seed} {arm} ({time.time() - t0:.0f}s)', flush=True)
