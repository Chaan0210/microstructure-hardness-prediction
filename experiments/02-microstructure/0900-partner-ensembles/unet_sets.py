"""Put the realistic-degradation U-Net (push-400 unet4-<NET>, phase_unet7.py) into the 038 / 039 / 044 variant sets for the NOISY rows only
(features-v3 noise > train median: the same label-free rule as the fine-tuned segmentation rows). Columns: the variant net columns
(10 m_* columns, het4_u, het4_u_mx), z_f25 / z_f50, and the two m_el_pix interactions (recomputed).
MODE replace: noisy rows take NET's values; avg: mean of the existing value and NET's value.
usage: unet_sets.py NET MODE SPLIT   -> feat-<v>{H2Z,H2Z-angh,-anghcor}-<NET><MODE>-{split}.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]
net, mode, split = sys.argv[1], sys.argv[2], sys.argv[3]
U = np.load(_repo_path(E, f'push-400/unet4-{net}-{split}.npz'), allow_pickle=True); uk = [str(k) for k in U['keys']]; uid = [str(i) for i in U['ids']]
Hh = np.load(_repo_path(E, f'push-500/uhet-u4{net}-{split}.npz'), allow_pickle=True); hk = [str(k) for k in Hh['keys']]; assert [str(i) for i in Hh['ids']] == uid
Ftr = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz')); thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))
F = np.load(_repo_path(E, f'grain-cellpose-220/features-v3-{split}.npz')); fpos = {str(i): k for k, i in enumerate(F['ids'])}; fnz = F['X'][:, list(F['keys']).index('noise')]
MCOLS = ['m_el_glob', 'm_el_loc16', 'm_el_pix', 'm_el_coh', 'm_la_std', 'm_la_f16', 'm_la_f32', 'm_la_tile8', 'm_la_mean', 'm_ph_hard']
new = {c: U['X'][:, uk.index(c)] for c in MCOLS + ['z_f25', 'z_f50']}; new.update({c: Hh['X'][:, hk.index(c)] for c in ('het4_u', 'het4_u_mx')})
sets = {'038': ('../push-500/feat-{v}H2Z-{s}.npz', 'feat-{v}H2Z-{t}-{s}.npz'), '039': ('../push-700/feat-{v}H2Z-angh-{s}.npz', 'feat-{v}H2Z-angh-{t}-{s}.npz'),
        '044': ('../push-800/feat-{v}-anghcor-{s}.npz', 'feat-{v}-anghcor-{t}-{s}.npz')}
for c, (src, dst) in sets.items():
    for v in (('u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z') if c == '044' else ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4')):
        S = np.load(_repo_path(H, src.format(v=v, s=split)), allow_pickle=True); sk = [str(k) for k in S['keys']]; X = S['X'].astype(float).copy(); sids = [str(i) for i in S['ids']]
        upos = [uid.index(i) for i in sids]; noisy = np.array([fnz[fpos[i]] > thr for i in sids])
        for col, val in new.items():
            if col not in sk: continue
            j = sk.index(col); nv = val[upos]; X[noisy, j] = nv[noisy] if mode == 'replace' else 0.5 * (X[noisy, j] + nv[noisy])
        for a, b in (('het4_u', 'm_el_pix'), ('sizefield_std_s32', 'm_el_pix')):
            if f'{a}_x_{b}' in sk: X[:, sk.index(f'{a}_x_{b}')] = X[:, sk.index(a)] * X[:, sk.index(b)]
        np.savez(_repo_path(H, dst.format(v=v, t=net + mode, s=split)), ids=S['ids'], X=X, keys=S['keys']); print(c, v, split, 'noisy rows', int(noisy.sum()), flush=True)
