"""P4: does a smooth illumination field blur the phase/matrix split even in CLEAN images?
Clean images (noise<7, zero-shot masks). Per-grain mean intensity; fit a smooth field (quadratic surface + Gaussian-kernel smoother, bw 48px)
to confident matrix grains (U-Net phase prob<0.15), subtract; compare darkness histograms (valley depth) and the information in the
corrected vs raw grain-threshold phase fraction (partial corr with y controlling other structure features; corr with 039 dev residual)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv
from scipy import ndimage as ndi
from PIL import Image
P = str(_REPO_ROOT) + '/'; E = P + 'experiments/'
U = np.load(_repo_io(E + 'push-400/unet4maps-z1-train.npz')); upos = {str(i): k for k, i in enumerate(U['ids'])}
Mc = np.load(_repo_io(E + 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); cpos = {str(i): k for k, i in enumerate(Mc['ids'])}
F = np.load(_repo_io(E + 'push-700/feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float); ids = [str(i) for i in F['ids']]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io(P + 'data/train.csv')))}
D = [np.load(_repo_io(E + f'push-600/dev-{v}H2Z-angh.npz'), allow_pickle=True) for v in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4')]
rid = [str(i) for i in D[0]['original_ids']]; res = dict(zip(rid, D[0]['original_y'] - np.mean([d['original_g'] for d in D], 0)))
nz = X[:, keys.index('noise')] * 255
rows = []; Hraw = np.zeros(40); Hcor = np.zeros(40); bins = np.arange(-20, 81, 2.5)
for k, i in enumerate(ids):
    if not (np.isfinite(nz[k]) and nz[k] < 7.5): continue
    im = np.asarray(Image.open(_repo_io(P + f'data/train/{i}.png')), np.float32); mask = Mc['masks'][cpos[i]].astype(int)
    mp = U['maps'][upos[i]].astype(np.float32); ph = np.kron(mp[1], np.ones((2, 2))); po = np.kron(mp[2], np.ones((2, 2)))
    inner = np.where(ndi.minimum_filter(mask, 5) == ndi.maximum_filter(mask, 5), mask, 0)
    labs = np.unique(inner); labs = labs[labs > 0]
    if len(labs) < 25: continue
    I = np.asarray(ndi.mean(im, inner, labs)); pph = np.asarray(ndi.mean(ph, inner, labs)); ppo = np.asarray(ndi.mean(po, inner, labs))
    A = np.asarray(ndi.sum(np.ones_like(im), mask, labs)); com = np.array(ndi.center_of_mass(np.ones_like(im), inner, labs)); cy, cx = com[:, 0], com[:, 1]
    keep = ppo < 0.3; I, pph, A, cx, cy = I[keep], pph[keep], A[keep], cx[keep], cy[keep]
    mat = pph < 0.15
    if mat.sum() < 15: continue
    # smooth field: Gaussian kernel regression over matrix grains (bw 48 px), leave-one-out for matrix grains themselves
    d2 = (cx[:, None] - cx[mat][None]) ** 2 + (cy[:, None] - cy[mat][None]) ** 2; Wk = np.exp(-d2 / (2 * 48 ** 2))
    selfm = np.where(mat)[0]; Wk[selfm, np.arange(mat.sum())] = 0
    field = (Wk @ I[mat]) / Wk.sum(1)
    d_raw = np.median(I[mat]) - I; d_cor = field - I
    Hraw += np.histogram(d_raw, bins, weights=A)[0]; Hcor += np.histogram(d_cor, bins, weights=A)[0]
    def frac(d, t): return A[d > t].sum() / A.sum()
    rows.append(dict(id=i, y=lab[i], res=res.get(i, np.nan), k=k, mat_sd_raw=np.std(d_raw[mat]), mat_sd_cor=np.std(d_cor[mat]),
                     **{f'raw{t}': frac(d_raw, t) for t in (12, 15, 18)}, **{f'cor{t}': frac(d_cor, t) for t in (12, 15, 18)},
                     soft=np.sum(A * pph) / A.sum()))
print('n clean images', len(rows), ' matrix-grain darkness SD raw %.2f -> field-corrected %.2f gray' % (np.median([r['mat_sd_raw'] for r in rows]), np.median([r['mat_sd_cor'] for r in rows])))
print('area-weighted darkness hist (raw | corrected):')
for a, h1, h2 in zip(bins[:-1], Hraw / Hraw.sum(), Hcor / Hcor.sum()):
    if -10 <= a <= 60: print(f'{a:6.1f} {"#" * int(h1 * 250):35s}| {"#" * int(h2 * 250)}')
y = np.array([r['y'] for r in rows]); rr = np.array([r['res'] for r in rows]); kk = np.array([r['k'] for r in rows])
ctrl = np.c_[[X[kk, keys.index(c)] for c in ('m_la_mean', 'het4_u', 'm_el_pix', 'ang_c4', 'pore_frac', 'n_obj')]].T
def pcorr(v, t, C):
    A_ = np.c_[np.ones(len(v)), C]; ev = v - A_ @ np.linalg.lstsq(A_, v, rcond=None)[0]; et = t - A_ @ np.linalg.lstsq(A_, t, rcond=None)[0]; return np.corrcoef(ev, et)[0, 1]
ok = np.isfinite(rr)
for c in ('soft', 'raw12', 'raw15', 'raw18', 'cor12', 'cor15', 'cor18'):
    v = np.array([r[c] for r in rows])
    print(f'{c:6s} partial corr with y | struct ctrl: {pcorr(v, y, ctrl):+.3f}   | struct + U-Net u_ph,z_f25,z_f50: {pcorr(v, y, np.c_[ctrl, X[kk][:, [keys.index("u_ph"), keys.index("z_f25"), keys.index("z_f50")]]]):+.3f}   corr with 039 resid {np.corrcoef(v[ok], rr[ok])[0, 1]:+.3f} (n={ok.sum()})')
