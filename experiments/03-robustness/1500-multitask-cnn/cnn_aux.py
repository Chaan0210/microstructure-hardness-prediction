"""EfficientNetV2-S (official torchvision ImageNet weights) fine-tuned for hardness WITH auxiliary structure targets: the network also regresses
14 standardized production measurements of each training image (phase fraction, block heterogeneity, elongation, size, darkness, cumulant phase
fraction, pores, orientation) -> label-informed representation learning on 320 images (train-analysis features as extra labels: allowed).
Grayscale -> 3ch, ImageNet norm; D4 + non-scaling crop 224 (+ brightness/contrast jitter); MSE(y) + AUXW * MSE(aux); AdamW (backbone LR, head 5x),
warmup+cosine, EMA; full-256 D4 TTA. Study900 dev folds (same ids/folds as every component). Nothing from test is used for fitting.
usage: cnn_aux.py PARTS FOLDS SEED EPOCHS LR AUXW TAG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, csv, copy, os, numpy as np, torch, torchvision
from pathlib import Path
from PIL import Image
parts = [int(x) for x in sys.argv[1].split(',')]; FOLDS = [int(x) for x in sys.argv[2].split(',')]; seed = int(sys.argv[3]); EP = int(sys.argv[4]); LR = float(sys.argv[5]); AUXW = float(sys.argv[6]); TAG = sys.argv[7]
P = Path(str(_REPO_ROOT)); E = _repo_path(P, 'experiments'); dev = torch.device('mps'); torch.set_num_threads(4)
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
sid = [str(i) for i in np.load(_repo_path(E, 'probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]
splits = [_repo_path(E, 'resolution-216/ensemble-OOF.npz'), _repo_path(E, 'resolution-repartition-217/20261031/ensemble-OOF.npz'), _repo_path(E, 'resolution-repartition-217/20261101/ensemble-OOF.npz')]
F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; fid = [str(i) for i in F['ids']]
C = np.load(_repo_path(E, 'push-1300/cumul-train.npz')); assert [str(i) for i in C['ids']] == fid
AUX = ['u_ph_hard', 'het4_u', 'het4_u_mx', 'm_el_pix', 'm_el_glob', 'm_la_mean', 'm_la_std', 'z_f50', 'sizefield_std_s32', 'n_obj', 'px_pore_frac', 'ang_c4', 'ang_s4']
A = np.c_[np.array([F['X'][:, fk.index(c)] for c in AUX]).T.astype(float), C['X'][:, 1]]; A[:, AUX.index('n_obj')] = np.log1p(np.nan_to_num(A[:, AUX.index('n_obj')], nan=0))
A = np.where(np.isfinite(A), A, np.nanmedian(A, 0)); AUXD = {i: A[j] for j, i in enumerate(fid)}; K = A.shape[1]
IM = {i: np.asarray(Image.open(_repo_path(P, f'data/train/{i}.png')).convert('L'), np.uint8) for i in sid}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32); CROP = 224
def to_t(batch):
    x = np.stack(batch).astype(np.float32) / 255.; x = np.repeat(x[:, None], 3, 1); return torch.from_numpy((x - MEAN[None, :, None, None]) / STD[None, :, None, None])
def build():
    m = torchvision.models.efficientnet_v2_s(); m.load_state_dict(torch.load(_repo_path(Path(__file__).resolve().parent, 'weights-dl/efficientnet_v2_s-dd5fe13b.pth'), map_location='cpu', weights_only=True))
    m.classifier = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(1280, 1 + K)); return m
def aug(a, rng):
    a = np.rot90(a, int(rng.integers(4)))
    if rng.random() < .5: a = a[:, ::-1]
    i, j = rng.integers(0, 256 - CROP + 1, 2); a = a[i:i + CROP, j:j + CROP].astype(np.float32)
    mu = a.mean(); a = mu + rng.uniform(0.85, 1.15) * (a - mu) + rng.uniform(-8, 8); return np.clip(a, 0, 255).astype(np.uint8)
@torch.no_grad()
def predict(m, ids):
    m.eval(); out = []
    for k in range(0, len(ids), 16):
        b = [IM[i] for i in ids[k:k + 16]]; acc = 0
        for r in range(4):
            for fl in (False, True):
                v = [np.ascontiguousarray(np.rot90(a, r)[:, ::-1] if fl else np.rot90(a, r)) for a in b]; acc = acc + m(to_t(v).to(dev))[:, 0].float().cpu().numpy()
        out.append(acc / 8)
    return np.concatenate(out)
fn = _repo_path(Path(__file__).resolve().parent, f'cnnaux-{TAG}-s{seed}.npz')
O = np.load(_repo_io(fn))['oof'] if fn.exists() else np.full((3, 400), np.nan)
for p_ in parts:
    d = np.load(_repo_io(splits[p_])); ids, fold = [str(i) for i in d['ids']], d['fold']
    for k in FOLDS:
        va = [i for i, f in zip(ids, fold) if f == k]
        if np.isfinite(O[p_, [sid.index(i) for i in va]]).all(): continue
        tr = [i for i, f in zip(ids, fold) if f != k]; y = np.array([lab[i] for i in tr], np.float32); mu, sdv = y.mean(), y.std(); yn = (y - mu) / sdv
        At = np.array([AUXD[i] for i in tr]); am, asd = At.mean(0), At.std(0) + 1e-6; An = ((At - am) / asd).astype(np.float32)
        torch.manual_seed(1000 * seed + 10 * p_ + k); rng = np.random.default_rng(1000 * seed + 10 * p_ + k)
        m = build().to(dev); ema = copy.deepcopy(m).eval()
        head = list(m.classifier.parameters()); hid = {id(q) for q in head}; body = [q for q in m.parameters() if id(q) not in hid]
        opt = torch.optim.AdamW([dict(params=body, lr=LR), dict(params=head, lr=5 * LR)], weight_decay=0.05)
        BS = 16; spe = int(np.ceil(len(tr) / BS)); tot = EP * spe; warm = 2 * spe
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, tot) / tot)))
        t0 = time.time(); step = 0
        for ep in range(EP):
            m.train(); perm = rng.permutation(len(tr))
            for b in range(spe):
                ix = perm[b * BS:(b + 1) * BS]
                if len(ix) < 4: continue
                x = to_t([aug(IM[tr[j]], rng) for j in ix]).to(dev); t = torch.from_numpy(yn[ix]).to(dev); ta = torch.from_numpy(An[ix]).to(dev)
                o = m(x); loss = torch.nn.functional.mse_loss(o[:, 0], t) + AUXW * torch.nn.functional.mse_loss(o[:, 1:], ta)
                opt.zero_grad(); loss.backward(); opt.step(); sch.step(); step += 1
                with torch.no_grad():
                    dcy = min(0.998, (1 + step) / (10 + step))
                    for pe, pm in zip(ema.parameters(), m.parameters()): pe.mul_(dcy).add_(pm.detach(), alpha=1 - dcy)
                    for be, bm in zip(ema.buffers(), m.buffers()): be.copy_(bm)
        pr = predict(ema, va) * sdv + mu; O[p_, [sid.index(i) for i in va]] = pr; yv = np.array([lab[i] for i in va])
        print(f'{TAG} seed {seed} part {p_} fold {k}: RMSE {np.sqrt(((pr - yv) ** 2).mean()):.3f}  ({time.time() - t0:.0f}s, {step} steps)', flush=True)
        np.savez(_repo_io(fn), oof=O)
