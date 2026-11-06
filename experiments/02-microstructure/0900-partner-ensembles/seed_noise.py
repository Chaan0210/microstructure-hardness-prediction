"""TabPFN seed noise of the final 044 slot (original features): 3 extra seeds (300, 400, 500) per variant vs the shipped 3 (0, 100, 200).
usage: seed_noise.py -> seed-noise-44.npz; prints RMS difference between the 3-seed means and the implied gain of 6 seeds."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
sys.path.insert(0, _repo_io('../push-300')); import harness
P0 = Path(_repo_io('../..')); lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P0, 'data/train.csv')))}
order = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data/sample_submission.csv')))]
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']; per = []
for v in V:
    Tr = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); Te = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-test.npz'), allow_pickle=True)
    pos = {str(i): k for k, i in enumerate(Te['ids'])}; Xte = Te['X'][[pos[i] for i in order]].astype(float); ytr = np.array([lab[str(i)] for i in Tr['ids']])
    ps = []
    for sd in (3, 4, 5):
        m = harness.tabpfn_new('v3.5', 100 * sd, 'mps'); m.fit(Tr['X'].astype(float), ytr); ps.append(m.predict(Xte))
    per.append(np.mean(ps, 0)); print(v, flush=True)
new = np.mean(per, 0); old = np.load(_repo_io('qm-final-tab44orig.npz'))['g']
np.savez(_repo_io('seed-noise-44.npz'), ids=np.array(order), g_extra=new, per=np.array(per))
d = new - old; print('RMS(3-seed mean A - B)', np.sqrt(np.mean(d ** 2)).round(3), ' -> seed noise variance of a 3-seed mean', (np.mean(d ** 2) / 2).round(3),
      ' MSE gain of 6 vs 3 seeds per slot ~', (np.mean(d ** 2) / 4).round(3))
