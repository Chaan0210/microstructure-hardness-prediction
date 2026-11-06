"""ZS-net: zone-level grain-size regression from degraded images (label-free; train images only).
Input: 128x128 window (a 64x64 zone + 32 px context, reflect-padded at the image border) of a degraded copy of a clean train image, normalised by
the full degraded image's 5/50/95 percentiles; D4 augmentation. Targets of the central zone from the CLEAN zero-shot maps:
  b    = -2 log( sum g a^-1/2 / sum g )   (Hall-Petch-type block statistic, as het4_u), g = matrix|phase pixels, a = area of containing grain
  lam  = sum g log a / sum g             (pixel-weighted mean log area)
  ph   = second-phase share of g pixels
  lcnt = log number of grains (>= 8 px) in the zone
Cross-fitting: FOLD A/B trains on one half of the clean images (fixed permutation, seed 0) so the other half's degraded copies are out-of-sample.
usage: zsnet.py train FOLD ITERS   (FOLD in A, B, all)   |   zsnet.py infer FOLDS SRC OUT   (SRC: train|test|synth:<set>)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, time, json, glob, numpy as np, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]
dev = torch.device('mps')
TN = ['b', 'lam', 'ph', 'lcnt']
def nstat(f):
    lo, md, hi = np.percentile(f, [5, 50, 95]); return md, max(hi - lo, 1.0)
def d4(x, k):
    if k & 1: x = np.swapaxes(x, -1, -2)
    if k & 2: x = x[..., ::-1, :]
    if k & 4: x = x[..., :, ::-1]
    return np.ascontiguousarray(x)
def zone_targets(T, LA, Mk, y0, x0):
    t = T[y0:y0 + 64, x0:x0 + 64]; la = LA[y0:y0 + 64, x0:x0 + 64]; g = (t == 0) | (t == 1)
    if g.sum() < 0.2 * 4096: return None
    b = -2 * np.log((np.exp(-0.5 * la) * g).sum() / g.sum()); lam = (la * g).sum() / g.sum(); ph = (t == 1).sum() / g.sum()
    lab = Mk[y0:y0 + 64, x0:x0 + 64]; c = np.bincount(lab.ravel()); c[0] = 0; lc = np.log(max((c >= 8).sum(), 1))
    return np.array([b, lam, ph, lc], np.float32)
def halves():
    D = np.load(_repo_path(H, 'zs-prep.npz')); n = len(D['ids']); p = np.random.RandomState(0).permutation(n); return np.sort(p[:n // 2]), np.sort(p[n // 2:]), n
class Zones(torch.utils.data.IterableDataset):
    def __init__(s, idx, bs, seed): s.idx, s.bs, s.seed = idx, bs, seed
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); r = np.random.RandomState(s.seed + (wi.id if wi else 0) * 7919)
        D = np.load(_repo_path(H, 'zs-prep.npz')); I, T, LA, M = D['images'], D['T'], D['LA'], D['masks']; bank = np.load(_repo_path(H, 'zs-bank-u8.npy'), mmap_mode='r')
        while True:
            xb = np.zeros((s.bs, 1, 128, 128), np.float32); tb = np.zeros((s.bs, 4), np.float32); b = 0
            while b < s.bs:
                k = int(r.choice(s.idx)); img = I[k] if r.rand() < 0.1 else bank[k, r.randint(bank.shape[1])]
                if r.rand() < 0.5: y0, x0 = 64 * r.randint(4), 64 * r.randint(4)
                else: y0, x0 = r.randint(0, 193, 2)
                t = zone_targets(T[k], LA[k], M[k], y0, x0)
                if t is None: continue
                f = img.astype(np.float32); md, sc = nstat(f); fp = np.pad((f - md) / sc, 32, mode='reflect')
                xb[b] = d4(fp[y0:y0 + 128, x0:x0 + 128], r.randint(8))[None]; tb[b] = t; b += 1
            yield torch.from_numpy(xb), torch.from_numpy(tb)
def net():
    m = torchvision.models.resnet50(); m.load_state_dict(torch.load(_repo_path(E, 'pretrained-cnn-185/weights/resnet50-11ad3fa6.pth'), map_location='cpu', weights_only=True))
    m.fc = nn.Linear(2048, 4); nn.init.zeros_(m.fc.weight); nn.init.zeros_(m.fc.bias); return m
def val_eval(model, mu, sd, idx):
    """aligned zones of 2 degraded copies of each held-out image: zone corr, within-image zone corr, image-level std corr."""
    D = np.load(_repo_path(H, 'zs-prep.npz')); T, LA, M = D['T'], D['LA'], D['masks']; bank = np.load(_repo_path(H, 'zs-bank-u8.npy'), mmap_mode='r')
    P, Y, G = [], [], []
    for k in idx:
        for j in (0, 1):
            img = bank[k, j].astype(np.float32); md, sc = nstat(img); fp = np.pad((img - md) / sc, 32, mode='reflect'); xs, ts = [], []
            for zy in range(4):
                for zx in range(4):
                    t = zone_targets(T[k], LA[k], M[k], 64 * zy, 64 * zx)
                    if t is None: continue
                    xs.append(fp[64 * zy:64 * zy + 128, 64 * zx:64 * zx + 128]); ts.append(t)
            if len(xs) < 4: continue
            with torch.no_grad(): p = model(torch.tensor(np.array(xs, np.float32))[:, None].repeat(1, 3, 1, 1).to(dev)).cpu().numpy() * sd + mu
            P.append(p); Y.append(np.array(ts)); G.append(np.full(len(ts), k * 2 + j))
    P, Y, G = np.concatenate(P), np.concatenate(Y), np.concatenate(G); out = {}
    for c, n in enumerate(TN):
        pw = P[:, c] - np.array([P[G == g, c].mean() for g in G]); yw = Y[:, c] - np.array([Y[G == g, c].mean() for g in G])
        sp = np.array([P[G == g, c].std() for g in np.unique(G)]); sy = np.array([Y[G == g, c].std() for g in np.unique(G)])
        out[n] = dict(zone=round(float(np.corrcoef(P[:, c], Y[:, c])[0, 1]), 3), within=round(float(np.corrcoef(pw, yw)[0, 1]), 3), img_std=round(float(np.corrcoef(sp, sy)[0, 1]), 3))
    return out
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'train':
        fold, iters = sys.argv[2], int(sys.argv[3]); A, B, n = halves(); trn = {'A': A, 'B': B, 'all': np.arange(n)}[fold]; val = {'A': B, 'B': A, 'all': B[:30]}[fold]
        torch.manual_seed(0)
        _, tt = next(iter(torch.utils.data.DataLoader(Zones(trn, 2000, 123), batch_size=None))); mu, sd = tt.mean(0), tt.std(0) + 1e-6
        model = net().to(dev); opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=iters, pct_start=0.05)
        dl = iter(torch.utils.data.DataLoader(Zones(trn, 64, 1), batch_size=None, num_workers=6, prefetch_factor=4, persistent_workers=True))
        t0 = time.time(); mud, sdd = mu.to(dev), sd.to(dev)
        for it in range(1, iters + 1):
            model.train(); x, t = next(dl); x, t = x.repeat(1, 3, 1, 1).to(dev), t.to(dev)
            loss = ((model(x) - (t - mud) / sdd) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if it % 500 == 0 or it == iters:
                model.eval(); print(json.dumps(dict(it=it, loss=round(float(loss), 4), sec=round(time.time() - t0), val=val_eval(model, mu.numpy(), sd.numpy(), val[:40]))), flush=True)
        torch.save(dict(state=model.state_dict(), mu=mu, sd=sd, fold=fold), _repo_path(H, f'zsnet-{fold}.pt')); print('DONE', flush=True)
    else:
        folds, src, out = sys.argv[2].split(','), sys.argv[3], sys.argv[4]
        if src.startswith('synth:'):
            files = sorted(glob.glob(str(_repo_path(E, f'push-400/synth/{src[6:]}/*.png')))); ids = [Path(f).stem for f in files]
        else:
            ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data', 'train.csv' if src == 'train' else 'sample_submission.csv')))]; files = [_repo_path(P0, 'data', src, f'{i}.png') for i in ids]
        models = []
        for f in folds:
            ck = torch.load(_repo_path(H, f'zsnet-{f}.pt'), map_location='cpu'); m = net(); m.load_state_dict(ck['state']); m.to(dev).eval(); models.append((m, ck['mu'].numpy(), ck['sd'].numpy()))
        R = np.zeros((len(ids), len(models), 16, 4), np.float32)
        for i, fn in enumerate(files):
            img = np.asarray(Image.open(_repo_io(fn))).astype(np.float32); md, sc = nstat(img); fp = np.pad((img - md) / sc, 32, mode='reflect')
            xs = np.array([d4(fp[64 * zy:64 * zy + 128, 64 * zx:64 * zx + 128], k) for zy in range(4) for zx in range(4) for k in range(8)])
            xt = torch.tensor(xs.astype(np.float32))[:, None].repeat(1, 3, 1, 1).to(dev)
            with torch.no_grad():
                for j, (m, mu, sd) in enumerate(models): R[i, j] = (m(xt).cpu().numpy() * sd + mu).reshape(16, 8, 4).mean(1)
            if i % 200 == 0: print(src, i, flush=True)
        np.savez(_repo_path(H, f'{out}.npz'), ids=np.array(ids), R=R, folds=np.array(folds), keys=np.array(TN)); print('saved', out, R.shape, flush=True)
