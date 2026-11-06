"""Geometry-regression CNN (label-free; train images only): ImageNet ResNet50 (local torchvision weights, public 2021) fine-tuned to recover the
13 exact geometry targets of clean-train-image crops (geom_prep.TNAMES, computed from the zero-shot mask) from degraded versions of the same crops
(geom-bank: phase_unet7 degradation mix) or the clean crop (10%). Random crop size per batch in {128..256}, D4 with orientation-target sign flips.
No hardness labels, no test images in training. 25 source images are held out for a label-free degraded-crop validation.
usage: geomnet.py train ITERS [tag]  |  geomnet.py infer [tag]   (-> geom-cnn<tag>-{train,test}.npz, full 256 image, 8-view D4 TTA)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, time, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(H)); from geom_prep import crop_targets, TNAMES
dev = torch.device('mps')
def norm(f):
    lo, med, hi = np.percentile(f, [5, 50, 95]); return ((f - med) / max(hi - lo, 1.0)).astype(np.float32)
def d4(x, k):
    if k & 1: x = np.swapaxes(x, -1, -2)
    if k & 2: x = x[..., ::-1, :]
    if k & 4: x = x[..., :, ::-1]
    return np.ascontiguousarray(x)
class Crops(torch.utils.data.IterableDataset):
    def __init__(s, idx, bs, seed): s.idx, s.bs, s.seed = idx, bs, seed
    def __iter__(s):
        wi = torch.utils.data.get_worker_info(); r = np.random.RandomState(s.seed + (wi.id if wi else 0) * 7919)
        D = np.load(_repo_path(H, 'geom-prep.npz')); I, M, T, G, off = D['images'], D['masks'], D['T'], D['G'], D['goff']; bank = np.load(_repo_path(H, 'geom-bank-u8.npy'), mmap_mode='r')
        while True:
            S = int(r.choice([128, 160, 192, 224, 256])); xb = np.zeros((s.bs, 3, S, S), np.float32); tb = np.zeros((s.bs, len(TNAMES)), np.float32)
            for b in range(s.bs):
                k = int(r.choice(s.idx)); img = I[k] if r.rand() < 0.1 else bank[k, r.randint(bank.shape[1])]
                y0, x0 = r.randint(0, 257 - S, 2); q = r.randint(8)
                f = d4(img[y0:y0 + S, x0:x0 + S].astype(np.float32), q); m = d4(M[k][y0:y0 + S, x0:x0 + S], q); t = d4(T[k][y0:y0 + S, x0:x0 + S], q)
                xb[b] = norm(f)[None]; tb[b] = crop_targets(m, t, G[off[k]:off[k + 1]], q)
            yield torch.from_numpy(xb), torch.from_numpy(tb)
def net():
    m = torchvision.models.resnet50(); m.load_state_dict(torch.load(_repo_path(E, 'pretrained-cnn-185/weights/resnet50-11ad3fa6.pth'), map_location='cpu', weights_only=True))
    m.fc = nn.Linear(2048, len(TNAMES)); nn.init.zeros_(m.fc.weight); nn.init.zeros_(m.fc.bias); return m
if __name__ == '__main__':
    mode = sys.argv[1]; tag = sys.argv[3] if mode == 'train' and len(sys.argv) > 3 else (sys.argv[2] if mode == 'infer' and len(sys.argv) > 2 else '')
    D0 = np.load(_repo_path(H, 'geom-prep.npz')); n = len(D0['ids']); rs = np.random.RandomState(0); val = rs.permutation(n)[:25]; trn = np.setdiff1d(np.arange(n), val)
    if mode == 'train':
        iters = int(sys.argv[2]); torch.manual_seed(0)
        mu_sd = next(iter(torch.utils.data.DataLoader(Crops(trn, 1500, 123), batch_size=None)))[1]; mu, sd = mu_sd.mean(0), mu_sd.std(0) + 1e-6
        model = net().to(dev); opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=iters, pct_start=0.05)
        dl = iter(torch.utils.data.DataLoader(Crops(trn, 32, 1), batch_size=None, num_workers=6, prefetch_factor=4, persistent_workers=True))
        vx, vt = next(iter(torch.utils.data.DataLoader(Crops(val, 400, 999), batch_size=None)))
        t0 = time.time(); mud, sdd = mu.to(dev), sd.to(dev)
        for it in range(1, iters + 1):
            model.train(); x, t = next(dl); x, t = x.to(dev), t.to(dev)
            loss = ((model(x) - (t - mud) / sdd) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if it % 250 == 0 or it == iters:
                model.eval()
                with torch.no_grad(): pv = torch.cat([model(vx[i:i + 50].to(dev)) for i in range(0, len(vx), 50)]).cpu() * sd + mu
                corr = [float(np.corrcoef(pv[:, j], vt[:, j])[0, 1]) for j in range(len(TNAMES))]
                print(json.dumps(dict(it=it, loss=round(loss.item(), 4), sec=round(time.time() - t0), val_corr={k: round(c, 3) for k, c in zip(TNAMES, corr)})), flush=True)
        torch.save(dict(state=model.state_dict(), mu=mu, sd=sd), _repo_path(H, f'geomnet{tag}.pt')); print('DONE', flush=True)
    else:
        ck = torch.load(_repo_path(H, f'geomnet{tag}.pt'), map_location='cpu', weights_only=False); model = net(); model.load_state_dict(ck['state']); model = model.to(dev).eval(); mu, sd = ck['mu'], ck['sd']
        for split in ('train', 'test'):
            ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data', 'train.csv' if split == 'train' else 'sample_submission.csv')))]; out = []
            with torch.no_grad():
                for i in ids:
                    f = norm(np.asarray(Image.open(_repo_path(P0, 'data', split, f'{i}.png')), np.float32)); xs = np.stack([np.repeat(d4(f, q)[None], 3, 0) for q in range(8)])
                    p = model(torch.from_numpy(xs).to(dev)).cpu() * sd + mu; p = p.numpy()
                    for q in range(8):   # undo the orientation sign flips of each view before averaging
                        if q & 1: p[q, 7] = -p[q, 7]
                        if q & 2: p[q, 8] = -p[q, 8]
                        if q & 4: p[q, 8] = -p[q, 8]
                    out.append(p.mean(0))
            np.savez(_repo_path(H, f'geom-cnn{tag}-{split}.npz'), ids=np.array(ids), X=np.array(out), keys=np.array(TNAMES)); print(split, 'done', flush=True)
