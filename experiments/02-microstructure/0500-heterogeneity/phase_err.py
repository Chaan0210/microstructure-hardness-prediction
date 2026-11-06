"""Where does the U-Net phase-fraction error come from under heavy degradation? Per-grain analysis on m1's 30 held-out val images."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, torch
from pathlib import Path
from scipy import ndimage as ndi
from skimage import measure
H = Path(_repo_io('../push-400')).resolve()
sys.argv = ['phase_unet2.py', 'chk', '1', '0']
src = open(_repo_path(H, 'phase_unet2.py')).read(); g = {'__name__': 'x', '__file__': str(_repo_path(H, 'phase_unet2.py'))}
exec(src.replace("if __name__ == '__main__':", "if False:"), g)
UNet, norm, imgs, TT, L, val, trn = [g[k] for k in ['UNet', 'norm', 'imgs', 'TT', 'L', 'val', 'trn']]
dev = torch.device('mps'); net = UNet(32).to(dev); net.load_state_dict(torch.load(_repo_path(H, 'unet2-m1.pt'), map_location=dev)); net.eval()
def deg(a, lab, r, off=5, blur=1.6, c=0.8, n=17):
    f = a.copy(); o = r.normal(0, off, lab.max() + 1); o[0] = 0; f = f + o[lab]
    f = ndi.gaussian_filter(f, blur); mu = f.mean(); f = mu + c * (f - mu)
    return np.clip(np.round(f + r.normal(0, n, f.shape)), 0, 255)
def pred(x):
    views = []
    for k in range(8):
        v = x
        if k & 1: v = v.T
        if k & 2: v = v[::-1]
        if k & 4: v = v[:, ::-1]
        views.append(np.ascontiguousarray(v))
    with torch.no_grad(): o = net(torch.tensor(np.stack(views))[:, None].to(dev))[:, :4].softmax(1).cpu().numpy()
    acc = np.zeros((4, 256, 256))
    for k in range(8):
        p = o[k]
        if k & 4: p = p[:, :, ::-1]
        if k & 2: p = p[:, ::-1]
        if k & 1: p = p.transpose(0, 2, 1)
        acc += p
    return acc / 8
r = np.random.RandomState(7); rows = []; fr = []
ks = list(val) + list(trn[:30])
for k in ks:
    T = TT[k]; m = L[k]; p = pred(norm(deg(imgs[k], m, r)))
    gm = T <= 1; ft = ((T == 1) & gm).sum() / gm.sum()
    fr.append((ft, p[1].sum() / (p[0] + p[1]).sum(), (p[1] > 0.5).sum() / (1 - p[2]).sum(), (p.argmax(0) == 1).sum() / (p.argmax(0) != 2).sum(), k in val))
    for rp in measure.regionprops(m):
        cls = np.bincount(T[m == rp.label], minlength=4)[:3].argmax()
        if cls == 2: continue
        inner = ndi.binary_erosion(m == rp.label, iterations=1)
        if inner.sum() < 3: continue
        rows.append((rp.area, cls, p[1][inner].mean(), k in val))
R_ = np.array(rows, float); fr = np.array(fr, float)
for nm, sel in (('val', fr[:, 4] == 1), ('train', fr[:, 4] == 0)):
    f = fr[sel]; print(nm, 'n', sel.sum(), 'corr soft %.3f gt05 %.3f argmax %.3f | rmse soft %.4f gt05 %.4f argmax %.4f' % (np.corrcoef(f[:, 0], f[:, 1])[0, 1], np.corrcoef(f[:, 0], f[:, 2])[0, 1], np.corrcoef(f[:, 0], f[:, 3])[0, 1], np.sqrt(np.mean((f[:, 0] - f[:, 1]) ** 2)), np.sqrt(np.mean((f[:, 0] - f[:, 2]) ** 2)), np.sqrt(np.mean((f[:, 0] - f[:, 3]) ** 2))))
v = R_[R_[:, 3] == 1]
for lo, hi in ((0, 50), (50, 150), (150, 400), (400, 1e9)):
    s = (v[:, 0] >= lo) & (v[:, 0] < hi)
    ph = s & (v[:, 1] == 1); mx = s & (v[:, 1] == 0)
    print(f'area {lo}-{hi}: phase grains {ph.sum()} mean p1 {v[ph, 2].mean():.3f} missed(<0.5) {np.mean(v[ph, 2] < 0.5):.3f} | matrix {mx.sum()} mean p1 {v[mx, 2].mean():.3f} false(>0.5) {np.mean(v[mx, 2] > 0.5):.3f} | area share of phase-missed {v[ph & (v[:, 2] < 0.5), 0].sum() / max(v[ph, 0].sum(), 1):.3f}')
