import sys, glob, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from PIL import Image
H = Path(__file__).resolve().parent; E = H.parent
stage, st = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(str(E / f'push-400/synth/train_{st}/*.png'))); ids = [Path(f).stem for f in files]
if stage == 'seg':
    sys.path.insert(0, str(E / 'push-300')); sys.path.insert(0, str(E / 'grain-cellpose-220'))
    import seglib
    from cellpose import models
    seglib._model = models.CellposeModel(gpu=True, pretrained_model=str(E / 'push-300/ft-H300.pt'))
    M = np.stack([seglib.seg(np.asarray(Image.open(f))) for f in files]).astype(np.uint16)
    np.savez_compressed(H / f'synth-H300-{st}.npz', ids=np.array(ids), masks=M); print('seg done', st, M.shape)
else:
    Z = np.load(H / f'synthmaps-train_{st}-u4z1.npz'); assert [str(i) for i in Z['ids']] == ids
    Mn = np.load(H / f'synth-H300-{st}.npz'); assert [str(i) for i in Mn['ids']] == ids
    sh = np.full(len(ids), np.nan)
    for k, f in enumerate(files):
        mask = Mn['masks'][k].astype(int); im = np.asarray(Image.open(f), np.float32)
        mp = Z['maps'][k].astype(np.float32); ph = np.kron(mp[1], np.ones((2, 2))); po = np.kron(mp[2], np.ones((2, 2)))
        inner = np.where(ndi.minimum_filter(mask, 5) == ndi.maximum_filter(mask, 5), mask, 0); labs = np.unique(inner); labs = labs[labs > 0]
        if len(labs) < 25: continue
        I = np.asarray(ndi.mean(im, inner, labs)); pph = np.asarray(ndi.mean(ph, inner, labs)); ppo = np.asarray(ndi.mean(po, inner, labs))
        A = np.asarray(ndi.sum(np.ones_like(im), mask, labs)); com = np.array(ndi.center_of_mass(np.ones_like(im), inner, labs)); cy, cx = com[:, 0], com[:, 1]
        keep = ppo < 0.3; I, pph, A, cx, cy = I[keep], pph[keep], A[keep], cx[keep], cy[keep]; mat = pph < 0.15
        if mat.sum() < 15: continue
        d2 = (cx[:, None] - cx[mat][None]) ** 2 + (cy[:, None] - cy[mat][None]) ** 2; W = np.exp(-d2 / (2 * 48 ** 2)); W[np.where(mat)[0], np.arange(mat.sum())] = 0
        field = (W @ I[mat]) / W.sum(1); sh[k] = A[field - I > 15].sum() / A.sum()
    acc = np.zeros((len(ids), 2))
    for n in ('m1', 'm2', 'h1', 'u4z1'):
        maps = np.load(H / f'synthmaps-train_{st}-{n}.npz')['maps']
        for k in range(len(ids)):
            m = maps[k].astype(np.float32); g = m[0] + m[1]; c2, s2 = m[5], m[6]; C = (g * c2).sum() / g.sum(); S = (g * s2).sum() / g.sum(); vv = np.hypot(C, S) + 1e-9
            acc[k] += [(C * C - S * S) / vv, 2 * C * S / vv]
    acc /= 4
    for v in ('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z'):
        S = np.load(H / f'synthrows-train_{st}-{v}.npz' if (H / f'synthrows-train_{st}-{v}.npz').exists() else E / f'push-500/synthrows-train_{st}-{v}.npz', allow_pickle=True); assert [str(i) for i in S['ids']] == ids; keys = [str(k) for k in S['keys']]; X = S['X'].astype(float)
        R = np.load(E / f'push-800/feat-{v}-anghcor-train.npz', allow_pickle=True); rk = [str(k) for k in R['keys']]
        new = np.c_[X, acc[:, 0], acc[:, 1], X[:, keys.index('het4_u')] * X[:, keys.index('m_el_pix')], X[:, keys.index('sizefield_std_s32')] * X[:, keys.index('m_el_pix')], sh]
        nk = keys + ['ang_c4', 'ang_s4', 'het4_u_x_m_el_pix', 'sizefield_std_s32_x_m_el_pix', 'sh_cor15']; assert nk == rk, (nk[-6:], rk[-6:])
        np.savez(H / f'synthrows-{st}-{v}-anghcor.npz', ids=np.array(ids), X=new, keys=np.array(nk))
    print('build done', st, 'sh nan', int(np.isnan(sh).sum()), 'sh mean %.4f' % np.nanmean(sh), 'ang sd', acc.std(0).round(3))
