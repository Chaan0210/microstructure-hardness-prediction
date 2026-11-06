"""Per-grain U-Net phase probability vs grain darkness z (distance from the z=0.3 rule) under heavy degradation, m1 val images.
usage: phase_err2.py NET [offset_sd]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, torch
from pathlib import Path
from scipy import ndimage as ndi
from skimage import measure
H = Path(_repo_io('../push-400')).resolve(); netname = sys.argv[1]; OFF = float(sys.argv[2]) if len(sys.argv) > 2 else 5
sys.argv = ['phase_unet2.py', 'chk', '1', '0']
src = open(_repo_path(H, 'phase_unet2.py')).read(); g = {'__name__': 'x', '__file__': str(_repo_path(H, 'phase_unet2.py'))}
exec(src.replace("if __name__ == '__main__':", "if False:"), g)
UNet, norm, imgs, TT, L, val = [g[k] for k in ['UNet', 'norm', 'imgs', 'TT', 'L', 'val']]
dev = torch.device('mps'); net = UNet(32).to(dev); net.load_state_dict(torch.load(_repo_path(H, f'unet2-{netname}.pt'), map_location=dev)); net.eval()
def deg(a, lab, r):
    f = a.copy(); o = r.normal(0, OFF, lab.max() + 1); o[0] = 0; f = f + o[lab]
    f = ndi.gaussian_filter(f, 1.6); mu = f.mean(); f = mu + 0.8 * (f - mu)
    return np.clip(np.round(f + r.normal(0, 17, f.shape)), 0, 255)
def pred(x):
    with torch.no_grad():
        vs = [x, x.T, x[::-1], x[:, ::-1]]; o = net(torch.tensor(np.stack([np.ascontiguousarray(v) for v in vs]))[:, None].to(dev))[:, :4].softmax(1).cpu().numpy()
    return (o[0] + o[1].transpose(0, 2, 1) + o[2][:, ::-1] + o[3][:, :, ::-1]) / 4
r = np.random.RandomState(7); rows = []; fr = []
for rep in range(2):
    for k in val:
        a = imgs[k]; m = L[k].astype(np.int64); T = TT[k]; p = pred(norm(deg(a, m, r)))
        gm = T <= 1; fr.append(((T == 1) & gm).sum() / gm.sum() - p[1].sum() / (p[0] + p[1]).sum())
        sm = ndi.gaussian_filter(a / 255.0, 1.0); inner = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3))
        props = {q.label: q for q in measure.regionprops(m)}; ipr = {q.label: q for q in measure.regionprops(inner, intensity_image=sm)}
        labs = np.array([l for l in props if l in ipr and ipr[l].area >= 3]); A = np.array([props[l].area for l in labs], float); I = np.array([ipr[l].intensity_mean for l in labs])
        o = np.argsort(I); c = np.cumsum(A[o]) / A.sum(); hi = I[o][np.searchsorted(c, 0.75)]; lo = np.percentile(sm, 0.1); z = (hi - I) / max(hi - lo, 1e-3)
        for l, zz, aa in zip(labs, z, A):
            cls = np.bincount(T[m == l], minlength=4)[:3].argmax()
            if cls == 2: continue
            inn = ndi.binary_erosion(m == l, iterations=1)
            if inn.sum() < 3: continue
            rows.append((aa, zz, p[1][inn].sum() / (p[0][inn] + p[1][inn]).sum()))
R_ = np.array(rows)
print(netname, 'off', OFF, 'phase frac bias (true-pred) mean %.4f sd %.4f' % (np.mean(fr), np.std(fr)))
for zl, zh in ((0.0, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 0.45), (0.45, 0.75)):
    s = (R_[:, 1] >= zl) & (R_[:, 1] < zh)
    out = []
    for al, ah in ((0, 80), (80, 200), (200, 1e9)):
        q = s & (R_[:, 0] >= al) & (R_[:, 0] < ah); out.append(f'A{al}-{int(min(ah, 9999))} n{q.sum():4d} p1 {R_[q, 2].mean() if q.any() else np.nan:.2f}')
    print(f'z {zl:.2f}-{zh:.2f}: ' + ' | '.join(out))
