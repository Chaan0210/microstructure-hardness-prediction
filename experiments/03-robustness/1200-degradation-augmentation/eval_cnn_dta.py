"""Evaluate cnn_dta.py outputs: seed-ensemble RMSE per arm and band, and the 065 dev blend with the CNN term moved by the paired difference
p021' = p021 + 1[noise >= GATE & ~failed] * SCALE * (mean_s dta_s - mean_s ctrl_s) (same weight 0.16; only partitions with both arms present).
usage: eval_cnn_dta.py [GATE=12] [SCALE=1]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, glob, numpy as np
D = np.load(_repo_io('dev065.npz'), allow_pickle=True); y, nz, fail, c = D['y'], D['nz'], D['fail'], D['c']
GATE = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0; SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
parts = ['original', '20261031', '20261101']; gate = (nz >= GATE) & ~fail
bands = [('clean', (nz < 9.5) & ~fail), ('mid', (nz >= 9.5) & (nz < 12) & ~fail), ('noisy', (nz >= 12) & ~fail)]
r = lambda e, m=slice(None): np.sqrt((e[m] ** 2).mean())
tot_old, tot_new = [], []
for k, p in enumerate(parts):
    A = {a: [np.load(_repo_io(f))['oof'] for f in sorted(glob.glob(f'cnn-{a}-{p}-s*.npz'))] for a in ('ctrl', 'dta')}
    n = min(len(A['ctrl']), len(A['dta']))
    if n == 0: continue
    ec, ed = np.mean(A['ctrl'][:n], 0), np.mean(A['dta'][:n], 0)
    print(f'{p} seeds {n}: single ctrl {np.mean([r(a - y) for a in A["ctrl"][:n]]):.3f} dta {np.mean([r(a - y) for a in A["dta"][:n]]):.3f} | ens ctrl {r(ec - y):.3f} dta {r(ed - y):.3f} | '
          + ' '.join(f'{b} {r(ec - y, m):.2f}->{r(ed - y, m):.2f}' for b, m in bands) + f' | p021 noisy {r(D["p021"][k] - y, bands[2][1]):.2f}')
    cn = c[k] + 0.16 * SCALE * gate * (ed - ec)
    print(f'   065 blend {r(c[k] - y):.4f} -> {r(cn - y):.4f}   noisy {r(c[k] - y, bands[2][1]):.3f} -> {r(cn - y, bands[2][1]):.3f}')
    tot_old.append(((c[k] - y) ** 2).mean()); tot_new.append(((cn - y) ** 2).mean())
if tot_old: print(f'partitions {len(tot_old)}: 065 {np.sqrt(np.mean(tot_old)):.4f} -> {np.sqrt(np.mean(tot_new)):.4f}')
