"""Calibrate realdeg2 on OUR train images only (label-free). Real statistics: clean rows (noise < 9.5; masks-v3) and noisy rows (noise >= 12;
ft-H300 1-view masks for mask-based statistics). Simulated: clean rows degraded by realdeg2 with noise drawn from the real noisy noise values,
mask-based statistics with the clean masks. Per image:
  noise_mad   white-noise SD from the MAD of the 2nd difference
  edge_sd     pore edge width: erf fit to the mean profile across pore boundaries (pores = compact components of sigma-1.5 darkness > 0.42)
  pore_z      pore core darkness (pixels >= 2.5 px inside pore boundaries; threshold-selected, biased)
  q99..q998, tail40  sigma-1.5 pixel darkness tail quantiles and share > 0.4 (pores, threshold-free)
  shade_amp   SD / mean of a quadratic surface fitted to bright pixels (sigma 3)
  dark15      share of sigma-3 pixels with darkness > 0.15;  q90, q95 pixel darkness quantiles
  mat_sd      SD of bright-grain means / S, corrected for pixel-noise averaging
  dip, seg_sd boundary dip / S (median over boundary pixels) and SD of per-segment mean dips (segments >= 15 px)
  ghist       area-weighted grain-darkness histogram (pooled)
usage: calib_own.py real | calib_own.py sim '{"MU_D":0.66,...}' [N_IMAGES]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi, special
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(H)); from realdeg2 import degrade2
YY, XX = np.mgrid[0:256, 0:256] / 255.0 - 0.5
Mc = np.load(_repo_path(E, 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); ids = [str(i) for i in Mc['ids']]
Mn = np.load(_repo_path(E, 'push-300/ftseg-ft-H300-train-noisy-n1-masks.npz'), allow_pickle=True); npos = {str(i): k for k, i in enumerate(Mn['ids'])}
nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255
GB = np.linspace(-0.1, 0.7, 33)

def surface(sm, step=2):
    s = sm[::step, ::step]; Y, X = YY[::step, ::step].ravel(), XX[::step, ::step].ravel(); v = s.ravel()
    D = np.c_[np.ones_like(X), X, Y, X * X, Y * Y, X * Y]; keep = v > np.quantile(v, 0.4)
    for _ in range(4): b = np.linalg.lstsq(D[keep], v[keep], rcond=None)[0]; keep = v / (D @ b) > 0.95
    Df = np.c_[np.ones(65536), XX.ravel(), YY.ravel(), XX.ravel() ** 2, YY.ravel() ** 2, (XX * YY).ravel()]
    return (Df @ b).reshape(256, 256)

def noise_mad(img):
    d = img[:, 2:] - 2 * img[:, 1:-1] + img[:, :-2]; return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(6)

def pore_stats(img):
    sm = ndi.gaussian_filter(img, 1.5); S = surface(sm); z = 1 - sm / S
    lab, k = ndi.label(z > 0.42)
    if k == 0: return np.nan, np.nan
    idx = np.arange(1, k + 1); area = np.bincount(lab.ravel())[1:]; mz = np.asarray(ndi.mean(z, lab, idx))
    sl = ndi.find_objects(lab); border = np.array([s[0].start == 0 or s[1].start == 0 or s[0].stop == 256 or s[1].stop == 256 for s in sl])
    good = idx[(area >= 25) & (mz > 0.5) & ~border]
    if len(good) == 0: return np.nan, np.nan
    mask = np.isin(lab, good); din = ndi.distance_transform_edt(mask); dout = ndi.distance_transform_edt(~mask)
    d = np.where(mask, -din + 0.5, dout - 0.5); v = img / S; bins = np.arange(-4, 7)
    sel = (d >= -4.5) & (d < 6.5); bi = np.clip(np.round(d[sel]).astype(int), -4, 6) + 4
    cnt = np.bincount(bi, minlength=11); prof = np.bincount(bi, weights=v[sel], minlength=11) / np.maximum(cnt, 1)
    core_bins = (bins <= -3) & (cnt > 5)
    if core_bins.sum() == 0: return np.nan, np.nan
    core = prof[core_bins].mean(); sur = prof[(bins >= 4) & (cnt > 5)].mean(); pn = (prof - core) / max(sur - core, 1e-6)
    best = (1e9, np.nan)
    for s in np.arange(0.3, 4.01, 0.05):
        for d0 in np.arange(-2, 2.01, 0.1):
            e = np.sum(cnt * (pn - special.ndtr((bins - d0) / s)) ** 2)
            if e < best[0]: best = (e, s)
    return float(best[1]), float(1 - core)

def tail_stats(img):
    sm = ndi.gaussian_filter(img, 1.5); S = surface(sm); z = (1 - sm / S)[3:-3, 3:-3].ravel()
    return float(np.quantile(z, 0.99)), float(np.quantile(z, 0.995)), float(np.quantile(z, 0.998)), float((z > 0.4).mean())

def pix_stats(img):
    sm = ndi.gaussian_filter(img, 3.0); S = surface(sm); z = (1 - sm / S)[4:-4:2, 4:-4:2].ravel()
    return float((z > 0.15).mean()), float(np.quantile(z, 0.9)), float(np.quantile(z, 0.95)), float(S.std() / S.mean())

def grain_stats(img, m, sdn):
    m = m.astype(np.int64); n = m.max() + 1; sm = ndi.gaussian_filter(img, 1.0)
    er = m * (ndi.minimum_filter(m, 3) == ndi.maximum_filter(m, 3)); cnt = np.bincount(er.ravel(), minlength=n).astype(float); w = np.maximum(cnt, 1)
    I = np.bincount(er.ravel(), weights=img.ravel(), minlength=n) / w; Ism = np.bincount(er.ravel(), weights=sm.ravel(), minlength=n) / w
    cy = np.bincount(er.ravel(), weights=YY.ravel(), minlength=n) / w; cx = np.bincount(er.ravel(), weights=XX.ravel(), minlength=n) / w
    ok = cnt >= 15; ok[0] = False
    if ok.sum() < 15: return None
    D = np.c_[np.ones(n), cx, cy, cx * cx, cy * cy, cx * cy]; keep = ok & (I > np.quantile(I[ok], 0.4))
    for _ in range(4):
        sw = np.sqrt(cnt[keep]); b = np.linalg.lstsq(D[keep] * sw[:, None], I[keep] * sw, rcond=None)[0]; S = D @ b; keep = ok & (I / S > 0.95)
    z = 1 - I / S; gh = np.histogram(z[ok], GB, weights=cnt[ok])[0] / cnt[ok].sum()
    mat = ok & (z < 0.08) & (cnt >= 60)
    mat_sd = np.sqrt(max(np.var(I[mat] / S[mat]) - np.mean(sdn ** 2 / (cnt[mat] * S[mat] ** 2)), 0)) if mat.sum() >= 8 else np.nan
    okb = cnt >= 30; okb[0] = False; dips = []; segs = {}
    for a, bb, v in ((m[:, :-1], m[:, 1:], (sm[:, :-1] + sm[:, 1:]) / 2), (m[:-1, :], m[1:, :], (sm[:-1, :] + sm[1:, :]) / 2)):
        s = (a != bb) & okb[a] & okb[bb]; Sm = (S[a[s]] + S[bb[s]]) / 2; dd = ((Ism[a[s]] + Ism[bb[s]]) / 2 - v[s]) / Sm; dips.append(dd)
        key = np.minimum(a[s], bb[s]) * n + np.maximum(a[s], bb[s]); segs.setdefault('k', []).append(key); segs.setdefault('d', []).append(dd)
    dips = np.concatenate(dips); key = np.concatenate(segs['k']); dd = np.concatenate(segs['d'])
    u, inv, c = np.unique(key, return_inverse=True, return_counts=True); sm_ = np.bincount(inv, weights=dd) / c; seg_sd = float(np.std(sm_[c >= 15])) if (c >= 15).sum() > 10 else np.nan
    return dict(mat_sd=mat_sd, dip=float(np.median(dips)), seg_sd=seg_sd, ghist=gh)

def stats(img, m, sdn):
    o = dict(noise_mad=noise_mad(img)); o['edge_sd'], o['pore_z'] = pore_stats(img)
    o['dark15'], o['q90'], o['q95'], o['shade_amp'] = pix_stats(img); o['q99'], o['q995'], o['q998'], o['tail40'] = tail_stats(img); g = grain_stats(img, m, sdn)
    if g: o.update(g)
    return o

KEYS = ['noise_mad', 'edge_sd', 'pore_z', 'shade_amp', 'dark15', 'q90', 'q95', 'q99', 'q995', 'q998', 'tail40', 'mat_sd', 'dip', 'seg_sd']
def summarize(rows):
    out = {}
    for k in KEYS:
        v = np.array([r.get(k, np.nan) for r in rows], float); v = v[np.isfinite(v)]
        out[k] = [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), int(len(v))]
    gh = np.array([r['ghist'] for r in rows if 'ghist' in r]); out['ghist'] = (gh.sum(0) / gh.sum()).tolist(); return out

if __name__ == '__main__':
    clean_idx = [n for n in range(len(ids)) if nz[n] < 9.5]; noisy_idx = [n for n in range(len(ids)) if nz[n] >= 12 and ids[n] in npos]
    if sys.argv[1] == 'real':
        res = {}
        mid_idx = [n for n in range(len(ids)) if 9.5 <= nz[n] < 12 and ids[n] in npos]
        for nm, idx in (('clean', clean_idx), ('noisy', noisy_idx), ('mid', mid_idx)):
            rows = []
            for n in idx:
                img = np.array(Image.open(_repo_path(P, f'data/train/{ids[n]}.png'))).astype(np.float64)
                m = Mc['masks'][n] if nm == 'clean' else Mn['masks'][npos[ids[n]]]; rows.append(stats(img, m, nz[n]))
            res[nm] = summarize(rows); print(nm, json.dumps({k: [round(x, 4) for x in res[nm][k][:3]] for k in KEYS}), flush=True)
        json.dump(res, open(_repo_path(H, 'calib-real.json'), 'w'), indent=1)
    else:
        prm = json.loads(sys.argv[2]); nimg = int(sys.argv[3]) if len(sys.argv) > 3 else len(clean_idx)
        real = json.load(open(_repo_path(H, 'calib-real.json'))); r = np.random.RandomState(7); import os
        lo, hi = float(os.environ.get('POOL_LO', 12)), float(os.environ.get('POOL_HI', 99)); mixp = float(os.environ.get('MIXP', 1.0)); tgt = os.environ.get('TARGET', 'noisy')
        LIGHT = dict(MU_D=1.0, SD_MUD=0.0, CD_SD=0.0, PORE_K=1.0, PORE_SD=0.0, J_MU=0.0, J_SD=0.0, P_LO=0.0, P_HI=0.0, BLUR_MU=0.0, BLUR_SD=0.0, SHADE_MAX=float(os.environ.get('LIGHT_SHADE', 0.0)))
        pool = nz[(nz >= lo) & (nz < hi)]; rows = []
        for n in clean_idx[:nimg]:
            img = np.array(Image.open(_repo_path(P, f'data/train/{ids[n]}.png'))).astype(np.float64); sdn = float(r.choice(pool))
            rows.append(stats(degrade2(img, Mc['masks'][n], r, prm if r.rand() < mixp else LIGHT, noise=sdn), Mc['masks'][n], sdn))
        s = summarize(rows); gr, gs = np.array(real[tgt]['ghist']), np.array(s['ghist'])
        print('PRM', json.dumps(prm)); print(f'{"stat":10s} {"clean":>22s} {tgt + " real":>22s} {"sim":>22s}')
        for k in KEYS: print(f'{k:10s} ' + ' '.join(f'{x[0]:7.4f} [{x[1]:.3f},{x[2]:.3f}]' for x in (real['clean'][k], real[tgt][k], s[k])))
        print(f'ghist L1 sim vs noisy real {np.abs(gr - gs).sum():.4f}  (clean vs noisy real {np.abs(np.array(real["clean"]["ghist"]) - gr).sum():.4f})')
        print('ghist cdf at z=0.1/0.15/0.2/0.3  real ' + ' '.join(f'{np.cumsum(gr)[np.searchsorted(GB, t) - 1]:.3f}' for t in (0.1, 0.15, 0.2, 0.3))
              + '  sim ' + ' '.join(f'{np.cumsum(gs)[np.searchsorted(GB, t) - 1]:.3f}' for t in (0.1, 0.15, 0.2, 0.3)), flush=True)
