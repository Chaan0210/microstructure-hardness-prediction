"""New variant set: feat-u1phdrop + 10 multi-task U-Net cols from one net (unet2-<NET> or unet4-<NET> for 'u4'-prefixed) + het4_u, het4_u_mx (that net's maps)
+ z cols from unet4-z1. Column order matches the H2Z sets. usage: make_variant.py NET OUT_TAG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
net, out = sys.argv[1], sys.argv[2]
cols = ['m_el_glob', 'm_el_loc16', 'm_el_pix', 'm_el_coh', 'm_la_std', 'm_la_f16', 'm_la_f32', 'm_la_tile8', 'm_la_mean', 'm_ph_hard']
for s in ('train', 'test'):
    F = np.load(_repo_io(f'../push-400/feat-u1phdrop-{s}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]
    U = np.load(_repo_io(f'../push-400/unet4-{net[2:]}-{s}.npz' if net.startswith('u4') else f'../push-400/unet2-{net}-{s}.npz'), allow_pickle=True); uk = [str(k) for k in U['keys']]
    Hh = np.load(_repo_io(f'uhet-{net}-{s}.npz'), allow_pickle=True); hk = [str(k) for k in Hh['keys']]
    Z = np.load(_repo_io(f'../push-400/unet4-z1-{s}.npz'), allow_pickle=True); zk = [str(k) for k in Z['keys']]
    for d in (U, Hh, Z): assert [str(i) for i in d['ids']] == ids
    X = np.hstack([F['X'].astype(float), U['X'][:, [uk.index(c) for c in cols]], Hh['X'][:, [hk.index('het4_u'), hk.index('het4_u_mx')]], Z['X'][:, [zk.index('z_f25'), zk.index('z_f50')]]])
    keys = [str(k) for k in F['keys']] + cols + ['het4_u', 'het4_u_mx', 'z_f25', 'z_f50']
    ref = np.load(_repo_io(f'feat-u1dm1szH2Z-{s}.npz'), allow_pickle=True); assert [str(k) for k in ref['keys']] == keys
    np.savez(_repo_io(f'feat-{out}-{s}.npz'), X=X, keys=np.array(keys), ids=np.array(ids)); print(out, s, X.shape)
