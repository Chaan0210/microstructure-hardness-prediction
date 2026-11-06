"""Generalised push-500/build_synth_rows3.py: H2Z variant rows for a synthetic set (cellpose features push-400/synth/feat-ft<SET>.npz, u1 features
push-400/unet-u1-synth<SET>.npz, multi-task/z U-Net features and maps from push-1200/synthfeat|synthmaps-<SET>-<net>.npz).
usage: build_rows_g.py SET   -> push-1200/synthrows-<SET>-<v>H2Z.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; st = sys.argv[1]
V = {'u1dm1sz': ['m1'], 'u1dm2sz': ['m2'], 'u1dmavg': ['m1', 'm2', 'h1'], 'u1dmavg4': ['m1', 'm2', 'h1', 'm3']}
def het(maps):
    out = []
    for m in maps:
        m = m.astype(np.float32); g = m[0] + m[1]; e = np.exp(-0.5 * m[4])
        def bstat(w):
            num = (w * e).reshape(4, 32, 4, 32).sum((1, 3)); den = w.reshape(4, 32, 4, 32).sum((1, 3)); ok = den > 0.2 * 32 * 32
            return (-2 * np.log(num[ok] / den[ok])).std()
        out.append([bstat(g), bstat(m[0])])
    return np.array(out)
A = np.load(_repo_path(E, f'push-400/synth/feat-ft{st}.npz'), allow_pickle=True); ak = [str(k) for k in A['keys']]
Uu = np.load(_repo_path(E, f'push-400/unet-u1-synth{st}.npz'), allow_pickle=True); uk = [str(k) for k in Uu['keys']]
sid = [str(i) for i in A['ids']]; assert sid == [str(i) for i in Uu['ids']]
Z = np.load(_repo_path(H, f'synthfeat-{st}-u4z1.npz'), allow_pickle=True); zk = [str(k) for k in Z['keys']]; assert [str(i) for i in Z['ids']] == sid
for v, nets in V.items():
    R = np.load(_repo_path(E, f'push-500/feat-{v}H2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in R['keys']]
    NF = [np.load(_repo_path(H, f'synthfeat-{st}-{n}.npz'), allow_pickle=True) for n in nets]; nk = [str(k) for k in NF[0]['keys']]
    assert all([str(i) for i in nf['ids']] == sid for nf in NF)
    HM = np.mean([het(np.load(_repo_path(H, f'synthmaps-{st}-{n}.npz'))['maps']) for n in nets], 0)
    cols = []
    for k in keys:
        if k in ('het4_u', 'het4_u_mx'): cols.append(HM[:, 0 if k == 'het4_u' else 1])
        elif k in ('z_f25', 'z_f50'): cols.append(Z['X'][:, zk.index(k)])
        elif k in nk and k.startswith('m_'): cols.append(np.mean([nf['X'][:, nk.index(k)] for nf in NF], 0))
        elif k in ak: cols.append(A['X'][:, ak.index(k)])
        elif k in uk: cols.append(Uu['X'][:, uk.index(k)])
        else: raise SystemExit(f'missing {k}')
    np.savez(_repo_path(H, f'synthrows-{st}-{v}H2Z.npz'), ids=np.array(sid), X=np.stack(cols, 1), keys=np.array(keys)); print(st, v, len(keys))
