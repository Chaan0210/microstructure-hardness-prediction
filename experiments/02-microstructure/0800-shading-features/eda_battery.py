"""EDA battery of a-priori image descriptors NOT in the 044 feature set (label-free), grouped by hypothesis.
Masks: zero-shot masks-v3 (noise <= train median) / ft-H300 1-view (noisier). U-Net z1 maps (128px, upsampled) for class probs and shape tensor.
usage: eda_battery.py SPLIT -> eda/battery-SPLIT.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
from pathlib import Path
from scipy import ndimage as ndi, stats
from skimage import measure
from PIL import Image
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]
split = sys.argv[1]
U = np.load(_repo_path(E, f'push-400/unet4maps-z1-{split}.npz')); upos = {str(i): k for k, i in enumerate(U['ids'])}
Mc = np.load(_repo_path(E, f'grain-cellpose-220/masks-v3-{split}.npz'), allow_pickle=True); cpos = {str(i): k for k, i in enumerate(Mc['ids'])}
Mn = np.load(_repo_path(E, f'push-300/ftseg-ft-H300-{split}-noisy-n1-masks.npz'), allow_pickle=True); npos = {str(i): k for k, i in enumerate(Mn['ids'])}
F = np.load(_repo_path(E, f'grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); fk = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
Ftr = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))
nz = F['X'][:, fk.index('noise')]
up = lambda a: np.kron(a, np.ones((2, 2)))
def neighbors(mask):
    """adjacency pairs between labels touching horizontally/vertically or across a 1-px boundary gap"""
    m = np.where(mask > 0, mask, ndi.grey_dilation(mask, size=3)) if (mask == 0).mean() > 0.01 else mask   # fill only zero gaps (label-number independent for real labels)
    pairs = set()
    for a, b in ((m[:, :-1], m[:, 1:]), (m[:-1, :], m[1:, :])):
        d = (a != b) & (a > 0) & (b > 0); pairs |= set(zip(np.minimum(a[d], b[d]).tolist(), np.maximum(a[d], b[d]).tolist()))
    return pairs
KEYS = None; rows = []
for n, i in enumerate(ids):
    o = {}
    noisy = nz[n] > thr
    mask = (Mn['masks'][npos[i]] if noisy else Mc['masks'][cpos[i]]).astype(int)
    im = np.asarray(Image.open(_repo_path(P, f'data/{split}/{i}.png')), np.float32)
    mp = U['maps'][upos[i]].astype(np.float32); pm, pph, ppo, pb = up(mp[0]), up(mp[1]), up(mp[2]), up(mp[3]); c2, s2 = up(mp[5]), up(mp[6])
    # --- H2 rendered levels and contrast-invariant ratios
    I_mat = np.median(im[pm > 0.8]) if (pm > 0.8).sum() > 50 else np.nan
    I_ph = np.median(im[pph > 0.8]) if (pph > 0.8).sum() > 50 else np.nan
    I_bnd = np.median(im[pb > 0.6]) if (pb > 0.6).sum() > 50 else np.nan
    I_po = np.percentile(im[ppo > 0.6], 10) if (ppo > 0.6).sum() > 20 else np.nan
    p01, p99 = np.percentile(im, [1, 99])
    o.update(I_mat=I_mat, rat_bnd=(I_mat - I_bnd) / max(I_mat - p01, 1), rat_ph=(I_mat - I_ph) / max(I_mat - p01, 1), rat_po=(I_mat - I_po) / max(I_mat - p01, 1),
             rat_bnd_ph=(I_mat - I_bnd) / max(I_mat - I_ph, 1) if np.isfinite(I_ph) else np.nan, bnd_frac=float((pb > 0.5).mean()))
    labs = np.unique(mask); labs = labs[labs > 0]
    if len(labs) >= 25:
        inner = np.where(ndi.minimum_filter(mask, 5) == ndi.maximum_filter(mask, 5), mask, 0)
        props = {p.label: p for p in measure.regionprops(mask)}
        L = np.array([l for l in labs if props[l].area >= 12])
        A = np.array([props[l].area for l in L], float); per = np.array([props[l].perimeter for l in L]); cen = np.array([props[l].centroid for l in L])
        AR = np.array([props[l].axis_major_length / max(props[l].axis_minor_length, 1) for l in L]); TH = np.array([props[l].orientation for l in L])
        cvx_per = np.array([measure.perimeter(props[l].convex_image) for l in L])
        I = np.asarray(ndi.mean(im, inner, L)); Ph = np.asarray(ndi.mean(pph, mask, L)); Po = np.asarray(ndi.mean(ppo, mask, L))
        isph = Ph > 0.5; ismat = (Ph < 0.15) & (Po < 0.1) & np.isfinite(I); ispo = Po > 0.5
        # perimeter-based boundary thickness: boundary-prob area / total grain-boundary length (each boundary shared by 2 grains)
        o['bnd_thick'] = float((pb > 0.5).sum() / max(per.sum() / 2, 1))
        o['straight'] = float(np.average(cvx_per / np.maximum(per, 1), weights=A))
        # --- H3 matrix brightness after smooth-field removal
        if ismat.sum() >= 20:
            cm = cen[ismat]; Im = I[ismat]; d2 = ((cm[:, None, :] - cm[None, :, :]) ** 2).sum(-1); W = np.exp(-d2 / (2 * 48 ** 2)); np.fill_diagonal(W, 0)
            r = Im - (W @ Im) / W.sum(1); sd = r.std()
            o.update(mb_sd=sd, mb_skew=stats.skew(r), mb_kurt=stats.kurtosis(r), mb_fbright=float(np.mean(r > 1.5 * sd)), mb_fdark=float(np.mean(r < -1.5 * sd)),
                     mb_corr_logA=stats.spearmanr(r, np.log(A[ismat]))[0], mb_corr_AR=stats.spearmanr(r, AR[ismat])[0],
                     mb_orient=float(np.hypot(np.corrcoef(r, np.cos(2 * TH[ismat]))[0, 1], np.corrcoef(r, np.sin(2 * TH[ismat]))[0, 1])),
                     mb_rel_ph=sd / max(I_mat - I_ph, 1) if np.isfinite(I_ph) else np.nan)
            Wn = (d2 < (1.3 * np.sqrt(65536 / len(labs))) ** 2) & (d2 > 0); a_, b_ = np.where(np.triu(Wn, 1))
            o['mb_nbr_corr'] = float(np.corrcoef(r[a_], r[b_])[0, 1]) if len(a_) > 10 else np.nan
        # --- H4 topology / regularity
        pr = neighbors(mask); deg = np.zeros(mask.max() + 1, int)
        for a, b in pr: deg[a] += 1; deg[b] += 1
        interior = np.array([props[l].bbox[0] > 3 and props[l].bbox[1] > 3 and props[l].bbox[2] < 253 and props[l].bbox[3] < 253 for l in L])
        dg = deg[L][interior]
        if len(dg) >= 10: o.update(frac_hex=float(np.mean(dg == 6)), deg_mean=float(dg.mean()), deg_cv=float(dg.std() / dg.mean()))
        zx = np.clip((cen[:, 1] // 64).astype(int), 0, 3); zy = np.clip((cen[:, 0] // 64).astype(int), 0, 3); zid = zy * 4 + zx
        cvs = [np.std(np.log(A[zid == z])) for z in range(16) if (zid == z).sum() >= 5]
        o['zone_within_logA_sd'] = float(np.mean(cvs)) if cvs else np.nan
        from scipy.spatial import cKDTree
        dnn = cKDTree(cen).query(cen, 2)[0][:, 1]; o['nn_cv'] = float(dnn.std() / dnn.mean())
        # --- H5 phase morphology
        if isph.sum() >= 5 and (~isph).sum() >= 5:
            Lidx = {l: k for k, l in enumerate(L)}; ph_l = set(L[isph].tolist())
            nb_ph = sum(1 for a, b in pr if a in ph_l and b in ph_l); nb_all = sum(1 for a, b in pr if a in Lidx and b in Lidx)
            f = isph.mean(); o['ph_cluster'] = float((nb_ph / max(nb_all, 1)) / max(f * f, 1e-6))
            o['ph_ar_ratio'] = float(np.average(AR[isph], weights=A[isph]) / np.average(AR[~isph], weights=A[~isph]))
            o['ph_count_frac'] = float(isph.mean()); o['ph_area_frac_grain'] = float(A[isph].sum() / A.sum())
            o['ph_size_cv'] = float(np.std(np.log(A[isph])))
        # --- H6 pores (U-Net pore prob components)
        lab_po, npo = ndi.label(ppo > 0.5)
        if npo >= 1:
            pp = measure.regionprops(lab_po); host = []; onb = []; al = []; sz = []
            for q in pp:
                if q.area < 6: continue
                cy, cx = map(int, q.centroid); ring = ndi.binary_dilation(lab_po == q.label, iterations=3) & (lab_po == 0)
                host.append(np.mean(pph[ring])); onb.append(np.mean(pb[ring] > 0.5)); sz.append(q.area)
                if q.axis_minor_length > 0: al.append(np.cos(2 * (q.orientation - np.arctan2(s2[cy, cx], c2[cy, cx]) / 2)) * (1 - q.axis_minor_length / q.axis_major_length))
            if sz:
                o.update(po_in_phase=float(np.mean(np.array(host) > 0.5)), po_on_bnd=float(np.mean(onb)), po_size_cv=float(np.std(np.log(sz))),
                         po_align=float(np.mean(al)) if al else np.nan, po_n=float(len(sz)))
    # --- H1 deformation field geometry (shape-tensor maps)
    g = pm + pph; vv = np.hypot(c2, s2)
    for s in (8, 32):
        cs = ndi.gaussian_filter(c2 * g, s) / np.maximum(ndi.gaussian_filter(g, s), 1e-3); ss = ndi.gaussian_filter(s2 * g, s) / np.maximum(ndi.gaussian_filter(g, s), 1e-3)
        th = 0.5 * np.arctan2(ss, cs); mag = np.hypot(cs, ss)
        gy, gx = np.gradient(np.exp(2j * th)); o[f'dir_grad_s{s}'] = float(np.average(np.abs(gy) + np.abs(gx), weights=mag + 1e-6))
        if s == 32:
            zm = mag.reshape(4, 64, 4, 64).mean((1, 3)); o['el_zone_cv'] = float(zm.std() / max(zm.mean(), 1e-6))
            zc = np.exp(2j * th).reshape(4, 64, 4, 64).mean((1, 3)); o['dir_zone_disp'] = float(1 - np.abs(zc.mean()) / max(np.abs(zc).mean(), 1e-6))
    # --- H7 spectrum (radially averaged log power slope, mid/high ratio)
    f2 = np.abs(np.fft.fftshift(np.fft.fft2(im - im.mean()))) ** 2; yy, xx = np.indices(f2.shape); rr = np.hypot(yy - 128, xx - 128).astype(int)
    rad = np.bincount(rr.ravel(), f2.ravel()) / np.maximum(np.bincount(rr.ravel()), 1); k = np.arange(4, 60)
    o['psd_slope'] = float(np.polyfit(np.log(k), np.log(rad[k] + 1e-9), 1)[0]); o['psd_mid_high'] = float(np.log(rad[8:24].sum() / max(rad[40:100].sum(), 1e-9)))
    if KEYS is None:
        KEYS = ['I_mat', 'rat_bnd', 'rat_ph', 'rat_po', 'rat_bnd_ph', 'bnd_frac', 'bnd_thick', 'straight', 'mb_sd', 'mb_skew', 'mb_kurt', 'mb_fbright', 'mb_fdark', 'mb_corr_logA',
                'mb_corr_AR', 'mb_orient', 'mb_rel_ph', 'mb_nbr_corr', 'frac_hex', 'deg_mean', 'deg_cv', 'zone_within_logA_sd', 'nn_cv', 'ph_cluster', 'ph_ar_ratio', 'ph_count_frac',
                'ph_area_frac_grain', 'ph_size_cv', 'po_in_phase', 'po_on_bnd', 'po_size_cv', 'po_align', 'po_n', 'dir_grad_s8', 'dir_grad_s32', 'el_zone_cv', 'dir_zone_disp', 'psd_slope', 'psd_mid_high']
    rows.append([float(o.get(k, np.nan)) for k in KEYS])
    if (n + 1) % 100 == 0: print(n + 1, flush=True)
(_repo_path(HERE, 'eda')).mkdir(exist_ok=True)
np.savez(_repo_path(HERE, f'eda/battery2-{split}.npz'), ids=np.array(ids), X=np.array(rows), keys=np.array(KEYS))
print('done', np.array(rows).shape, 'nan frac', np.isnan(np.array(rows)).mean(0).round(2))
