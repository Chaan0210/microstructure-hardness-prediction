"""[h9s2 vs h9] Holdout-100 check of H9: TabPFN members (3 slots x 4 variants, seeds 0,1,2) with +2 cumulant columns, fit on the 400 dev rows, predicting the holdout
rows; nb+h9 used on non-gated rows, full+h9 on gated rows; compared with holdout-featdrop.npz (noboth = nb, full). -> holdout-h9.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300')); import harness
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
dev = set(str(i) for i in np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']); H = np.load(_repo_io('holdout-featdrop.npz')); H9 = np.load(_repo_io('holdout-h9.npz')); hid = [str(i) for i in H['hid']]; out = {'hid': np.array(hid)}
for s in ('038', '039', '044'):
    for v in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4'):
        for fs in ('nb', 'full'):
            F = np.load(_repo_io(f'abl/feat-h9s2-{fs}-{s}-{v}-train.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; X = F['X'].astype(float)
            tr = np.array([i in dev for i in ids]); assert [ids[j] for j in np.where(~tr)[0]] == hid; y = np.array([lab[i] for i in ids]); ps = []
            for sd in (0, 1, 2): m = harness.tabpfn_new('v3.5', 100 * sd + 7, 'mps'); m.fit(X[tr], y[tr]); ps.append(m.predict(X[~tr]))
            out[f'{s}_{v}_{fs}'] = np.mean(ps, 0)
    print(s, flush=True)
np.savez(_repo_io('holdout-h9s2.npz'), **out)
y = np.array([lab[i] for i in hid]); nz = H['noise']; ng = nz < 12; S = ['038', '039', '044']; V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
b0 = np.mean([np.where(ng, H9[f'{s}_{v}_nb'], H9[f'{s}_{v}_full']) for s in S for v in V], 0); b9 = np.mean([np.where(ng, out[f'{s}_{v}_nb'], out[f'{s}_{v}_full']) for s in S for v in V], 0)
for nm, m in (('non-gated', ng), ('gated', ~ng), ('all', np.ones(len(y), bool))):
    d = (b9 - y)[m] ** 2 - (b0 - y)[m] ** 2; print(f'holdout block {nm} (n={m.sum()}): {np.sqrt(((b0 - y)[m] ** 2).mean()):.3f} -> {np.sqrt(((b9 - y)[m] ** 2).mean()):.3f}  dMSE {d.mean():+.2f} ± {d.std() / np.sqrt(m.sum()):.2f}')
