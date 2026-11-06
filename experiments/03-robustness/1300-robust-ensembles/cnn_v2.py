"""EfficientNetV2 (torchvision, official ImageNet-1k weights from weights-dl/) fine-tuned end-to-end for hardness, Study900 dev folds (same ids/folds as
every other component). Grayscale 256x256 replicated to 3 channels, ImageNet normalisation. Augmentation: D4 (rot90 x flip) + random 224 crop WITHOUT
rescaling (grain size is label information). Loss MSE on standardised y; AdamW, cosine with warmup, BN in train mode, EMA of weights.
Prediction: full 256 image, D4 8-view TTA, EMA weights. Nothing from test is used.
usage: cnn_v2.py ARCH(s|m) PARTS(0 | 0,1,2) SEED [EPOCHS] [LR]  -> cnnv2-<ARCH>-s<SEED>.npz (oof [3,400], NaN where not run)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, csv, copy, numpy as np, torch, torchvision
from pathlib import Path
from PIL import Image
import os
arch, parts, seed = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], int(sys.argv[3]); EP = int(sys.argv[4]) if len(sys.argv) > 4 else 60; LR = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-4
RES = int(os.environ.get('RES', 256)); CROP = RES * 7 // 8; FREEZE = int(os.environ.get('FREEZE', 0)); FOLDS = [int(x) for x in os.environ.get('FOLDS', '0,1,2,3,4').split(',')]; TAG = os.environ.get('TAG', '')
P = Path(_repo_io('../..')); dev = torch.device('mps'); torch.set_num_threads(4)
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
sys.path.insert(0, _repo_io('../push-900')); import numpy as _np
sid = [str(i) for i in np.load(_repo_path(P, 'experiments/probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]
splits = [_repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')]
IM = {i: np.asarray(Image.open(_repo_path(P, f'data/train/{i}.png')).convert('L').resize((RES, RES), Image.Resampling.BOX) if RES != 256 else Image.open(_repo_path(P, f'data/train/{i}.png')).convert('L'), np.uint8) for i in sid}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
def to_t(batch):   # list of HxW uint8 -> N,3,H,W normalised
    x = np.stack(batch).astype(np.float32) / 255.; x = np.repeat(x[:, None], 3, 1); x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]; return torch.from_numpy(x)
def build():
    if arch == 's': m = torchvision.models.efficientnet_v2_s(); sd = torch.load(_repo_io('weights-dl/efficientnet_v2_s-dd5fe13b.pth'), map_location='cpu', weights_only=True)
    else: m = torchvision.models.efficientnet_v2_m(); sd = torch.load(_repo_io('weights-dl/efficientnet_v2_m-dc08266a.pth'), map_location='cpu', weights_only=True)
    m.load_state_dict(sd)
    for blk in list(m.features)[:FREEZE]:
        for q in blk.parameters(): q.requires_grad_(False)
    nf = m.classifier[1].in_features; m.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(nf, 1)); return m
def aug(a, rng):
    a = np.rot90(a, int(rng.integers(4)))
    if rng.random() < .5: a = a[:, ::-1]
    i, j = rng.integers(0, RES - CROP + 1, 2); return np.ascontiguousarray(a[i:i + CROP, j:j + CROP])
@torch.no_grad()
def predict(m, ids):
    m.eval(); out = []
    for k in range(0, len(ids), 16):
        b = [IM[i] for i in ids[k:k + 16]]; acc = 0
        for r in range(4):
            for fl in (False, True):
                v = [np.ascontiguousarray(np.rot90(a, r)[:, ::-1] if fl else np.rot90(a, r)) for a in b]; acc = acc + m(to_t(v).to(dev)).float().cpu().numpy().ravel()
        out.append(acc / 8)
    return np.concatenate(out)
fn = f'cnnv2-{arch}{TAG}-s{seed}.npz'
O = np.load(_repo_io(fn))['oof'] if Path(fn).exists() else np.full((3, 400), np.nan)
for p_ in parts:
    d = np.load(_repo_io(splits[p_])); ids, fold = [str(i) for i in d['ids']], d['fold']
    for k in FOLDS:
        va = [i for i, f in zip(ids, fold) if f == k]
        if np.isfinite(O[p_, [sid.index(i) for i in va]]).all(): continue
        tr = [i for i, f in zip(ids, fold) if f != k]; y = np.array([lab[i] for i in tr], np.float32); mu, sdv = y.mean(), y.std(); yn = (y - mu) / sdv
        torch.manual_seed(1000 * seed + 10 * p_ + k); rng = np.random.default_rng(1000 * seed + 10 * p_ + k)
        m = build().to(dev); ema = copy.deepcopy(m).eval(); opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=LR, weight_decay=0.05)
        BS = 16; spe = int(np.ceil(len(tr) / BS)); tot = EP * spe; warm = 2 * spe
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, tot) / tot)))
        t0 = time.time(); step = 0
        for ep in range(EP):
            m.train(); perm = rng.permutation(len(tr))
            for b in range(spe):
                ix = perm[b * BS:(b + 1) * BS]
                if len(ix) < 4: continue
                x = to_t([aug(IM[tr[j]], rng) for j in ix]).to(dev); t = torch.from_numpy(yn[ix]).to(dev)
                loss = torch.nn.functional.mse_loss(m(x).squeeze(1), t); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); step += 1
                with torch.no_grad():
                    dcy = min(0.998, (1 + step) / (10 + step))
                    for pe, pm in zip(ema.parameters(), m.parameters()): pe.mul_(dcy).add_(pm.detach(), alpha=1 - dcy)
                    for be, bm in zip(ema.buffers(), m.buffers()): be.copy_(bm)
        pr = predict(ema, va) * sdv + mu; O[p_, [sid.index(i) for i in va]] = pr; yv = np.array([lab[i] for i in va])
        print(f'{arch} seed {seed} part {p_} fold {k}: RMSE {np.sqrt(((pr - yv) ** 2).mean()):.3f}  ({time.time() - t0:.0f}s, {step} steps)', flush=True)
        np.savez(_repo_io(fn), oof=O)
