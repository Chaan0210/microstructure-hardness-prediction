"""Pixel-weighted Hall-Petch-type block statistics from multi-task U-Net maps (label-free, all images, noise-robust).
b_j = -2 log( sum_p g(p) exp(-la(p)/2) / sum_p g(p) ) over block j, g = p_matrix + p_phase, la = predicted log area of containing grain.
het4_u / het8_u = std of b_j over 4x4 / 8x8 blocks; hp_u = global b; also b-range and phase-split versions.
usage: unet_het.py NET[,NET...]   -> uhet-<NET>-{train,test}.npz (features averaged over NETs if several)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; U = _repo_path(H.parent, 'push-400')
nets = sys.argv[1].split(',')
KEYS = ['het4_u', 'het8_u', 'hp_u', 'rng4_u', 'het4_u_mx', 'het2_u', 'lg4_std', 'lg4_max', 'la_skew']
def feats(m):
    g = m[0] + m[1]; e = np.exp(-0.5 * m[4]); out = {}
    def bstat(nb, w):
        s = 128 // nb; num = (w * e).reshape(nb, s, nb, s).sum((1, 3)); den = w.reshape(nb, s, nb, s).sum((1, 3)); ok = den > 0.2 * s * s
        return -2 * np.log(num[ok] / den[ok])
    b4 = bstat(4, g); b8 = bstat(8, g); b2 = bstat(2, g)
    out['het4_u'] = b4.std(); out['het8_u'] = b8.std(); out['het2_u'] = b2.std(); out['rng4_u'] = b4.max() - b4.min()
    out['hp_u'] = -2 * np.log((g * e).sum() / g.sum())
    out['het4_u_mx'] = bstat(4, m[0]).std()
    den = g.reshape(4, 32, 4, 32).sum((1, 3)); ok = den > 0.2 * 32 * 32; lg = (g * m[4]).reshape(4, 32, 4, 32).sum((1, 3))[ok] / den[ok]
    out['lg4_std'] = lg.std(); out['lg4_max'] = lg.max() - lg.mean()
    sel = g > 0.5; la = m[4][sel]; out['la_skew'] = float(((la - la.mean()) ** 3).mean() / max(la.std(), 1e-6) ** 3) if sel.sum() > 10 else 0.0
    return [out[k] for k in KEYS]
for split in ('train', 'test'):
    D = [dict(np.load(_repo_path(U, f'unet4maps-{n[2:]}-{split}.npz' if n.startswith('u4') else f'unet2maps-{n}-{split}.npz'))) for n in nets]
    ids = D[0]['ids']; assert all((d['ids'] == ids).all() for d in D)
    X = np.mean([[feats(d['maps'][i].astype(np.float32)) for i in range(len(ids))] for d in D], 0)
    np.savez(_repo_path(H, f'uhet-{"+".join(nets)}-{split}.npz'), ids=ids, X=X, keys=np.array(KEYS)); print(split, X.shape, flush=True)
