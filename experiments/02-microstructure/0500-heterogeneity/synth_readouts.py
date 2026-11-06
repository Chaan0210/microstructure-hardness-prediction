"""Label-free calibration data: degrade clean TRAIN images (noise < median) K times with a real-noisy-like recipe, run U-Net z1 (D4 TTA),
store phase/z/size readouts of degraded copies + the same readouts on the clean original (target). usage: synth_readouts.py K SEED"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, json, numpy as np, torch
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
H = Path(__file__).resolve().parent; U = _repo_path(H.parent, 'push-400'); P = _RepoPath(__file__).resolve().parents[3]
K, SEED = int(sys.argv[1]), int(sys.argv[2]); PHF = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0; NSUB = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9; OFFSD = float(sys.argv[5]) if len(sys.argv) > 5 else -1; CORR = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
src = open(_repo_path(U, 'phase_unet4.py')).read(); g = {}
exec('import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\n' + src[src.index('def blk'):src.index("if __name__ == '__main__':")], g)
dev = torch.device('mps'); net = g['UNet'](32).to(dev); net.load_state_dict(torch.load(_repo_path(U, 'unet4-z1.pt'), map_location=dev)); net.eval()
LA_MU, LA_SD = 5.636784553527832, 0.9691801071166992
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]; nz = np.load(_repo_path(H.parent, 'push-300/noise_train.npy')) * 255; med = np.median(nz)
M = np.load(_repo_path(H.parent, 'grain-cellpose-220/masks-v3-train.npz'))['masks']; pool = np.sort(nz[nz >= med])
def norm(f): lo, md, hi = np.percentile(f, [5, 50, 95]); return ((f - md) / max(hi - lo, 1.0)).astype(np.float32)
def predict(a):
    x = norm(a); vs = []
    for k in range(8):
        v = x
        if k & 1: v = v.T
        if k & 2: v = v[::-1]
        if k & 4: v = v[:, ::-1]
        vs.append(np.ascontiguousarray(v))
    with torch.no_grad():
        o = net(torch.tensor(np.stack(vs))[:, None].to(dev)); pr = o[:, :4].softmax(1).cpu().numpy(); la = (o[:, 4] * LA_SD + LA_MU).cpu().numpy(); zz = (o[:, 7] * 0.15 + 0.15).cpu().numpy()
    acc = np.zeros((6, 256, 256))
    for k in range(8):
        p = np.concatenate([pr[k], la[k][None], zz[k][None]])
        if k & 4: p = p[:, :, ::-1]
        if k & 2: p = p[:, ::-1]
        if k & 1: p = p.transpose(0, 2, 1)
        acc += p
    return acc / 8
KEYS = ['soft', 'argmax', 'gt05', 'unc', 'z_f20', 'z_f25', 'z_f30', 'z_f50', 'z_mean', 'la_mean', 'bnd', 'noise']
def readout(p, img):
    p0, p1, p2, p3, la, zm = p; gg = p0 + p1; am = p[:4].argmax(0); o = {}
    o['soft'] = p1.sum() / gg.sum(); o['argmax'] = (am == 1).sum() / max(((am == 0) | (am == 1)).sum(), 1); o['gt05'] = (p1 > 0.5).mean() / (1 - p2).mean()
    o['unc'] = ((p1 > 0.2) & (p1 < 0.7)).mean()
    for t in (0.2, 0.25, 0.3, 0.5): o[f'z_f{int(round(t * 100))}'] = (gg / (1 + np.exp(-(zm - t) / 0.02))).sum() / gg.sum()
    o['z_mean'] = (gg * zm).sum() / gg.sum(); o['la_mean'] = (gg * la).sum() / gg.sum(); o['bnd'] = p3.mean()
    f = img / 255; hh = (f[0::2, 0::2] - f[0::2, 1::2] - f[1::2, 0::2] + f[1::2, 1::2]) / 2; o['noise'] = float(np.median(np.abs(hh)) / 0.6745 * 255)
    return [float(o[k]) for k in KEYS]
yy, xx = np.mgrid[0:256, 0:256] / 255.0 - 0.5
clean = np.where(nz < med)[0][:NSUB]; T = []; D = []; who = []
for c, k in enumerate(clean):
    a = np.array(Image.open(_repo_path(P, f'data/train/{ids[k]}.png'))).astype(np.float64); lab = M[k].astype(np.int64)
    T.append(readout(predict(a), a))
    for rep in range(K):
        r = np.random.RandomState(SEED * 100000 + k * 10 + rep)
        off = r.normal(0, r.uniform(0, 6) if OFFSD < 0 else OFFSD, lab.max() + 1); off[0] = 0
        if PHF < 1.0:   # phase-contrast compression: grain means darker than the matrix level move toward it (pores excluded)
            Ig = ndi.mean(a, lab, np.arange(lab.max() + 1)); Ag = np.bincount(lab.ravel(), minlength=lab.max() + 1).astype(float); Ig[0] = np.nan
            o_ = np.argsort(np.nan_to_num(Ig, nan=-1)); valid = ~np.isnan(Ig); srt = np.argsort(Ig[valid]); Iv = Ig[valid][srt]; Av = Ag[valid][srt]; Im = Iv[np.searchsorted(np.cumsum(Av) / Av.sum(), 0.75)]
            lo_ = np.percentile(ndi.gaussian_filter(a, 1.0), 0.1); zg = (Im - Ig) / max(Im - lo_, 1e-3)
            sh_ = np.where(valid & (Ig < Im) & (zg < 0.7), (1 - PHF) * (Im - Ig), 0.0); sh_[0] = 0; off = off + sh_
        f = a + off[lab]
        cq = r.normal(0, 1, 5); sh = cq[0] * yy + cq[1] * xx + cq[2] * yy * yy + cq[3] * xx * xx + cq[4] * xx * yy; f = f + r.uniform(0, 6) * sh / max(sh.std(), 1e-6)
        sb = r.uniform(0.6, 2.0); f = ndi.gaussian_filter(f, sb); cc = r.uniform(0.65, 0.95); mu = f.mean(); f = mu + cc * (f - mu) + r.uniform(-10, 10)
        tgt = pool[r.randint(len(pool))]; exist = nz[k] * cc / (2 * np.sqrt(np.pi) * sb); add = np.sqrt(max(tgt ** 2 - exist ** 2, 1.0))
        if CORR > 0:   # part of the noise variance as spatially correlated (gaussian-filtered) noise, same total power budget split
            cn = ndi.gaussian_filter(r.normal(0, 1, f.shape), CORR); cn = cn / cn.std(); f = f + 0.5 * add * cn; add = add * np.sqrt(0.75)
        f = np.clip(np.round(f + r.normal(0, add, f.shape)), 0, 255)
        D.append(readout(predict(f), f)); who.append(c)
    if (c + 1) % 50 == 0: print(c + 1, flush=True)
np.savez(_repo_path(H, f'synth-readouts-K{K}-s{SEED}-phf{PHF}-off{OFFSD}-c{CORR}.npz'), T=np.array(T), D=np.array(D), who=np.array(who), ids=np.array([ids[k] for k in clean]), keys=np.array(KEYS)); print('done')
