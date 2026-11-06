"""Measurement-level calibration of the clean->noisy degradation: for each parameter setting, degrade 100 clean train sources (noise from the real
>= 12 pool), run the production phase U-Net u1 (D4 TTA, same normalisation) and compare the marginal distribution of its features with the real
noisy train rows (microstructure distributions are equal across bands, so a correct rendering must reproduce the measured-feature marginals).
Distance = sum over features of standardised mean difference^2 + log SD-ratio^2; pixel-statistic distance reported alongside. Label-free, train only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, itertools, numpy as np, torch
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, _repo_io('../push-900')); from realdeg2 import degrade2
exec(open(_repo_io('../push-1200/unet_feats_sg.py')).read().split("dev = torch.device")[0].split("name, split = sys.argv[1], sys.argv[2]")[1].replace('import importlib.util', '').split("spec =")[0])
src_defs = open(_repo_io('../push-1200/unet_feats_sg.py')).read(); exec(src_defs[src_defs.index('import torch.nn as nn'):src_defs.index('dev = torch.device')])
dev = torch.device('mps'); net = UNet().to(dev); net.load_state_dict(torch.load(_repo_io('../push-400/unet-u1.pt'), map_location=dev)); net.eval()
exec(src_defs[src_defs.index('def tta(x):'):src_defs.index('X = []; maps')])
def feats(a):
    p = tta(norm(a.astype(np.float32))); g = p[0] + p[1]; am = p.argmax(0)
    f = ndi.gaussian_filter(p[1], 32) / np.maximum(ndi.gaussian_filter(g, 32), 1e-3)
    return [p[1].sum() / max(g.sum(), 1), (am == 1).sum() / max(((am == 0) | (am == 1)).sum(), 1), np.log(max(p[3].mean(), 1e-4)), f.std()]
FK = ['u_ph', 'u_ph_hard', 'u_logbnd', 'u_ph_f32']
U = np.load(_repo_io('../push-400/unet-u1-train.npz'), allow_pickle=True); uk = [str(k) for k in U['keys']]; RU = U['X'][:, [uk.index(k) for k in FK]].astype(float)
exec(open(_repo_io('imgstats.py')).read().split("names = [")[0])
assert [str(i) for i in U["ids"]] == ids
import torch.nn.functional as F
real = RU[nz >= 12]; mu, sd = real.mean(0), real.std(0); M = np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'))['masks']; pool = np.sort(nz[nz >= 12])
PS = np.load(_repo_io('pixstats.npz')); J5 = [1, 2, 6, 3, 4]; prl = PS['train_X'][nz >= 12][:, J5]; pmu, psd = prl.mean(0), prl.std(0)
srcs = np.where(nz < np.median(nz))[0][::2][:100]
print('real noisy u1 means', np.round(mu, 3), 'sds', np.round(sd, 3), '| clean-band means', np.round(RU[nz < 9.5].mean(0), 3), flush=True)
def run(prm):
    FF, PSX = [], []
    for k in srcs:
        r = np.random.RandomState(7_000 + k); a = np.asarray(Image.open(_repo_io(f'{P}/data/train/{ids[k]}.png')).convert('L'), float)
        tgt = float(pool[r.randint(len(pool))]); add = float(np.sqrt(max(tgt ** 2 - (0.4 * nz[k]) ** 2, 1.0)))
        g = degrade2(a, M[k], r, prm, noise=add); FF.append(feats(g)); Image.fromarray(g.astype(np.uint8)).save(_repo_io('_q.png')); PSX.append(np.array(stats('_q.png'))[J5])
    FF = np.array(FF); PSX = np.array(PSX); smd = (FF.mean(0) - mu) / sd; lsr = np.log(FF.std(0) / sd)
    return float((smd ** 2).sum() + (lsr ** 2).sum()), float(np.sqrt((((PSX.mean(0) - pmu) / psd) ** 2).sum())), smd, lsr
grid = list(itertools.product([0.027, 0.05, 0.08], [0.675, 0.55, 0.8], [(0.28, 0.68), (0.8, 1.0)], [0.75, 0.9]))
res = []
for J, mud, (plo, phi), bl in grid:
    dU, dP, smd, lsr = run(dict(J_MU=J, J_SD=J * 0.3, MU_D=mud, P_LO=plo, P_HI=phi, BLUR_MU=bl)); res.append((dU, dP, J, mud, plo, bl))
    print(f'J {J:.3f} mu_d {mud:.3f} P {plo:.2f}-{phi:.2f} blur {bl:.2f}: U-dist {dU:6.2f} pix-dist {dP:.2f} | SMD {np.round(smd, 2)} logSDratio {np.round(lsr, 2)}', flush=True)
res.sort(); print('BEST by U-dist', res[:3]); print('BEST by U+pix', sorted(res, key=lambda t: t[0] + t[1] ** 2)[:3])
import os; os.remove('_q.png')
