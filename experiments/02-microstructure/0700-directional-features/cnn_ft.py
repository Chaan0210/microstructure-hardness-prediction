"""Full fine-tune of an ImageNet CNN (torchvision convnext_tiny / efficientnet_b0, local weights) on raw 256x256 images for hardness.
Official dev partition folds (train 320 / val 80). Augmentation: random 90-degree rotations only (C4 keeps cos4θ and sin4θ of the elongation direction).
AdamW (backbone LR, head LR x10), cosine with warmup, standardized-target MSE; C4 TTA at evaluation. Label-free inputs; no test images.
usage: cnn_ft.py ARCH PARTITION EPOCHS [LR]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, numpy as np, torch, torch.nn as nn, torchvision
from PIL import Image
arch, part, EPOCHS = sys.argv[1], sys.argv[2], int(sys.argv[3]); LR = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-4
dev = torch.device('mps'); torch.manual_seed(0)
D = np.load(_repo_io('../push-600/dev-u1dmavgH2Z-angh.npz'), allow_pickle=True)
ids = [str(i) for i in D[part + '_ids']]; y = D[part + '_y'].astype(np.float32); fold = D[part + '_fold']
IM = np.stack([np.array(Image.open(_repo_io(f'../../data/train/{i}.png'))).astype(np.float32) / 255.0 for i in ids])
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
def build():
    if arch == 'convnext':
        m = torchvision.models.convnext_tiny(weights=None); m.load_state_dict(torch.load(_repo_io('../modern-backbones-190/weights/convnext_tiny-983f1562.pth'), map_location='cpu'))
        m.classifier[2] = nn.Linear(768, 1); head = list(m.classifier[2].parameters())
    else:
        m = torchvision.models.efficientnet_b0(weights=None); m.load_state_dict(torch.load(_repo_io('../pretrained-cnn-185/weights/efficientnet_b0_rwightman-7f5810bc.pth'), map_location='cpu'))
        m.classifier[1] = nn.Linear(1280, 1); head = list(m.classifier[1].parameters())
    hid = {id(p) for p in head}; body = [p for p in m.parameters() if id(p) not in hid]
    return m.to(dev), body, head
def prep(x): x = x[:, None].repeat(1, 3, 1, 1); return (x - MEAN) / STD
def predict(m, X):
    m.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(X), 16):
            xb = torch.tensor(X[b:b + 16]).to(dev); out.append(torch.stack([m(prep(torch.rot90(xb, k, (1, 2))))[:, 0] for k in range(4)], 0).mean(0).cpu().numpy())
    return np.concatenate(out)
oof = np.zeros(len(y)); log = []
for k in range(5):
    t0 = time.time(); tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
    mu, sd = y[tr].mean(), y[tr].std(); yt = (y - mu) / sd
    m, body, head = build(); opt = torch.optim.AdamW([{'params': body, 'lr': LR}, {'params': head, 'lr': LR * 10}], weight_decay=0.05)
    bs = 16; steps = EPOCHS * int(np.ceil(len(tr) / bs)); warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    rng = np.random.RandomState(100 + k); step = 0
    for ep in range(EPOCHS):
        m.train(); perm = rng.permutation(tr); tl = []
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]; xb = torch.tensor(IM[j]).to(dev); kr = rng.randint(4); xb = torch.rot90(xb, kr, (1, 2))
            loss = ((m(prep(xb))[:, 0] - torch.tensor(yt[j]).to(dev)) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1; tl.append(loss.item())
        if ep % 10 == 9 or ep == EPOCHS - 1:
            pv = predict(m, IM[va]) * sd + mu; print(json.dumps(dict(fold=k, ep=ep + 1, train_loss=round(float(np.mean(tl)), 4), val_rmse=round(float(np.sqrt(np.mean((pv - y[va]) ** 2))), 3), sec=round(time.time() - t0))), flush=True)
    oof[va] = predict(m, IM[va]) * sd + mu
    del m, opt; torch.mps.empty_cache()
np.save(_repo_io(f'cnn-{arch}-e{EPOCHS}-{part}.npy'), oof)
g = np.mean([np.load(_repo_io(f'../push-600/dev-{v}-angh.npz'), allow_pickle=True)[part + '_g'] for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']], 0)
p021 = D[part + '_p021']; rm = lambda v: float(np.sqrt(np.mean((v - y) ** 2)))
print('DONE', arch, 'oof rmse %.3f | 039 grain %.3f | errcorr(039) %.3f | errcorr(021) %.3f' % (rm(oof), rm(g), np.corrcoef(oof - y, g - y)[0, 1], np.corrcoef(oof - y, p021 - y)[0, 1]))
