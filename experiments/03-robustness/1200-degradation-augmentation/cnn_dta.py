"""Degradation-transfer augmentation for the raw-image CNN (dev, Study900 partitions): TinyCNN128 with the legacy recipe (cnn_controls.TinyCNN,
BOX128 /255, train-only normalisation, D4, MSE, AdamW 2e-3/wd 1e-2, cosine 80 epochs, clip 5, mean of epoch 76-80 states).
arm 'ctrl': legacy batches of 32 real rows (800 updates).  arm 'dta': the same real batch + 32 label-preserving degraded copies of clean train
images (measured noise >= 12, source not in the validation fold) per update.  Both arms share init/order streams per (seed, fold).
usage: cnn_dta.py PART(original|20261031|20261101) SEEDS(comma) ARMS(comma) [SETS]  -> cnn-<arm>-<part>-s<seed>.npz (validation preds, Study900 order)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, time, numpy as np, torch
from pathlib import Path
from PIL import Image
P = Path(_repo_io('../..')); sys.path.insert(0, str(_repo_path(P, 'studies'))); from cnn_controls import TinyCNN; sys.path.pop(0)
F = torch.nn.functional; dev = 'mps' if torch.backends.mps.is_available() else 'cpu'; torch.set_num_threads(2)
part, seeds, arms = sys.argv[1], [int(s) for s in sys.argv[2].split(',')], sys.argv[3].split(',')
SETS = (sys.argv[4] if len(sys.argv) > 4 else 'deg,deg2,deg3,deg4,deg5,deg6,deg7').split(','); NMIN = float(os.environ.get('NOISEMIN', '12'))
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'),
          '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
lab = {r.split(',')[0]: float(r.split(',')[1]) for r in open(_repo_path(P, 'data/train.csv')).read().split()[1:]}
img = lambda f: np.asarray(Image.open(_repo_io(f)).resize((128, 128), Image.Resampling.BOX), dtype=np.float32) / 255.
d = np.load(_repo_io(splits[part])); ids = [str(i) for i in d['ids']]; y = d['y'].astype(np.float64); fold = d['fold']
X = np.stack([img(_repo_path(P, 'data/train', f'{i}.png')) for i in ids])[:, None]
sid = [str(i) for i in np.load(_repo_path(P, 'experiments/probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]; jj = [ids.index(i) for i in sid]
SX, SID = [], []
for s in SETS:
    S = np.load(_repo_io(f'synthrows-{s}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); k = [str(x) for x in S['keys']]; nz = S['X'][:, k.index('noise')].astype(float) * 255
    for i, n in zip(S['ids'], nz):
        if n >= NMIN: SX.append(img(_repo_path(P, f'experiments/push-400/synth/train_{s}/{i}.png'))); SID.append(str(i))
SX = np.stack(SX)[:, None]; SY = np.array([lab[i] for i in SID]); SID = np.array(SID); print(part, 'copies', len(SID), flush=True)
def d4(xb, rng):
    xb = np.rot90(xb, int(rng.integers(4)), axes=(2, 3)); return np.ascontiguousarray(xb[:, :, :, ::-1] if rng.random() < .5 else xb)
@torch.no_grad()
def predict(m, x, nm):
    m.eval(); return np.concatenate([m(torch.from_numpy(np.ascontiguousarray((x[j:j + 64] - nm[0]) / nm[1])).to(dev)).cpu().numpy() for j in range(0, len(x), 64)]) * nm[3] + nm[2]
for seed in seeds:
    for arm in arms:
        out = Path(f'cnn-{arm}-{part}-s{seed}.npz')
        if out.exists(): continue
        oof = np.zeros(400); t0 = time.time()
        for k in range(5):
            T, V = np.flatnonzero(fold != k), np.flatnonzero(fold == k); vids = set(np.array(ids)[V]); keep = np.flatnonzero([s not in vids for s in SID])
            nm = (float(X[T].mean()), float(X[T].std()), float(y[T].mean()), float(y[T].std()))
            XT = (X[T] - nm[0]) / nm[1]; yT = (y[T] - nm[2]) / nm[3]; XS = (SX[keep] - nm[0]) / nm[1]; yS = (SY[keep] - nm[2]) / nm[3]
            torch.manual_seed(1000 * seed + k); m = TinyCNN().to(dev); o = torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=80, eta_min=.0001); rng = np.random.default_rng(1000 * seed + k); rs = np.random.default_rng(7 + 1000 * seed + k); avg = None
            for ep in range(80):
                order = rng.permutation(len(T))
                for f0 in range(0, len(T), 32):
                    ix = order[f0:f0 + 32]; xb = d4(XT[ix], rng); yb = yT[ix]
                    if arm == 'dta':
                        js = rs.integers(len(keep), size=32); xb = np.concatenate([xb, d4(XS[js], rs)]); yb = np.r_[yb, yS[js]]
                    m.train(); o.zero_grad(set_to_none=True); loss = F.mse_loss(m(torch.from_numpy(xb).to(dev)), torch.tensor(yb, dtype=torch.float32, device=dev))
                    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.); o.step()
                sch.step()
                if ep >= 75:
                    st = {n: v.detach().cpu().clone() for n, v in m.state_dict().items()}
                    avg = {n: v / 5 for n, v in st.items()} if avg is None else {n: avg[n] + st[n] / 5 for n in st}
            m.load_state_dict(avg); oof[V] = predict(m, X[V], nm)
        np.savez(_repo_io(out), oof=oof[jj], ids=np.array(sid)); e = oof - y
        print(f'{part} s{seed} {arm}: rmse {np.sqrt((e ** 2).mean()):.3f}  ({time.time() - t0:.0f}s)', flush=True)
