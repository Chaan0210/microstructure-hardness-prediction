"""Probe: does hardness depend on WHERE in the image the structure is? Regional (k x k grid) U-Net phase fraction / log-area / het vs hardness."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv, sys
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
M = np.load(_repo_io('../push-400/unet4maps-z1-train.npz'), allow_pickle=True); maps = M['maps']; ids = [str(i) for i in M['ids']]
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; X = np.nan_to_num(F['X'])
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; y = np.array([lab[i] for i in ids])
s_ = 128 // K
PH = np.zeros((500, K * K)); LA = np.zeros((500, K * K)); ZF = np.zeros((500, K * K))
for n in range(500):
    m = maps[n].astype(np.float32); g = m[0] + m[1]; z = m[7]; zf = g / (1 + np.exp(-(z - 0.25) / 0.02))
    for a in range(K):
        for b in range(K):
            sl = (slice(a * s_, (a + 1) * s_), slice(b * s_, (b + 1) * s_)); gg = g[sl].sum()
            PH[n, a * K + b] = m[1][sl].sum() / gg; LA[n, a * K + b] = (g[sl] * m[4][sl]).sum() / gg; ZF[n, a * K + b] = zf[sl].sum() / gg
np.savez(_repo_io(f'region{K}-train.npz'), PH=PH, LA=LA, ZF=ZF)
nz = X[:, keys.index('noise')] * 255
for nm, s in [('HQ', nz < 11.9), ('all', nz > -1)]:
    for fn, R in [('phase', PH), ('zf25', ZF), ('logA', LA)]:
        # regress y on regional values + other-block global controls
        ctrl = ['het4_u', 'm_la_std', 'm_el_pix', 'm_el_glob', 'sizefield_std_s32'] + (['m_la_mean'] if fn != 'logA' else ['u_ph', 'z_f25'])
        C = np.c_[np.ones(s.sum()), X[s][:, [keys.index(k) for k in ctrl]]]
        A_ = np.c_[C, R[s]]; c = np.linalg.lstsq(A_, y[s], rcond=None)[0][C.shape[1]:]
        # bootstrap SE
        rng = np.random.RandomState(0); B = []
        idx = np.where(s)[0]
        for _ in range(200):
            bi = rng.choice(idx, len(idx)); Ab = np.c_[np.ones(len(bi)), X[bi][:, [keys.index(k) for k in ctrl]], R[bi]]; B.append(np.linalg.lstsq(Ab, y[bi], rcond=None)[0][C.shape[1]:])
        se = np.std(B, 0)
        print(nm, fn, 'region coefs (row-major):'); print('   ', np.round(c.reshape(K, K), 1).tolist(), ' se~', np.round(se.mean(), 1))
