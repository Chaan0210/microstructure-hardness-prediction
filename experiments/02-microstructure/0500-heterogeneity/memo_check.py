"""Memorization check: U-Net m1 phase-fraction / logA recovery on degraded copies of its TRAINING source images vs its 30 held-out val images."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, torch
from pathlib import Path
H = Path(_repo_io('../push-400')).resolve()
sys.argv = ['phase_unet2.py', 'chk', '1', '0']
src = open(_repo_path(H, 'phase_unet2.py')).read()
g = {'__name__': 'x', '__file__': str(_repo_path(H, 'phase_unet2.py'))}
exec(src.replace("if __name__ == '__main__':", "if False:"), g)
UNet, degrade, norm, imgs, TT, RR, VV, L, trn, val = [g[k] for k in ['UNet', 'degrade', 'norm', 'imgs', 'TT', 'RR', 'VV', 'L', 'trn', 'val']]
dev = torch.device('mps'); net = UNet(32).to(dev); net.load_state_dict(torch.load(_repo_path(H, 'unet2-m1.pt'), map_location=dev)); net.eval()
LA_MU, LA_SD = 5.636784553527832, 0.9691801071166992
from scipy import ndimage as ndi
def deg(a, lab, r, lvl):
    f = a.copy()
    off = r.normal(0, lvl['off'], lab.max() + 1); off[0] = 0; f = f + off[lab]
    f = ndi.gaussian_filter(f, lvl['blur']); mu = f.mean(); f = mu + lvl['c'] * (f - mu)
    return np.clip(np.round(f + r.normal(0, lvl['n'], f.shape)), 0, 255)
levels = {'heavy_nooff': dict(off=0, blur=1.6, c=0.75, n=18), 'heavy_noblur': dict(off=5, blur=0.0, c=0.75, n=18), 'heavy_lownoise': dict(off=5, blur=1.6, c=0.75, n=6)}
def run(ks, lvl, seed):
    r = np.random.RandomState(seed); pt, pp, lt, lp = [], [], [], []
    for k in ks:
        x = norm(deg(imgs[k], L[k], r, lvl))
        with torch.no_grad():
            o = net(torch.tensor(x)[None, None].to(dev))[0]; pr = o[:4].softmax(0).cpu().numpy(); la = (o[4] * LA_SD + LA_MU).cpu().numpy()
        T = TT[k]; gm = T <= 1; pt.append(((T == 1) & gm).sum() / gm.sum()); pp.append(pr[1].sum() / (pr[0] + pr[1]).sum())
        V = VV[k]; lt.append(RR[k][0][V].mean()); lp.append(la[V].mean())
    pt, pp, lt, lp = map(np.array, (pt, pp, lt, lp))
    return np.corrcoef(pt, pp)[0, 1], np.sqrt(np.mean((pt - pp) ** 2)), np.corrcoef(lt, lp)[0, 1]
for name, lvl in levels.items():
    for grp, ks in [('val', val)]:
        res = [run(ks, lvl, s) for s in (1, 2)]
        print(name, grp, len(ks), 'phase corr %.4f rmse %.4f logA corr %.4f' % tuple(np.mean(res, 0)), flush=True)
