"""Test predictions of the nb (no het/intensity) TabPFN block: per slot, mean over 4 U-Net variants x seeds (random_state 100*sd, sd 0,1,2),
fit on all 500 train rows (production finalize_multi.py procedure). Also the full-feature 044 slot to check reproduction against the slot
recovered from candidate 044's CSV. usage: final_nb_tab.py DEVICE -> fin-nb-tab.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300')); import harness
dev = sys.argv[1]; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]; V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
def member(ftr, fte):
    Tr = np.load(_repo_io(ftr), allow_pickle=True); Te = np.load(_repo_io(fte), allow_pickle=True); assert [str(i) for i in Te['ids']] == order and list(Tr['keys']) == list(Te['keys'])
    X, y, Xe = Tr['X'].astype(float), np.array([lab[str(i)] for i in Tr['ids']]), Te['X'].astype(float); ps = []
    for sd in (0, 1, 2): m = harness.tabpfn_new('v3.5', 100 * sd, dev); m.fit(X, y); ps.append(m.predict(Xe))
    return np.mean(ps, 0)
out = {}
for s in ('038', '039', '044'):
    out[f'nb_{s}'] = np.mean([member(f'abl/feat-nb-{s}-{v}-train.npz', f'abl/feat-nb-{s}-{v}-test.npz') for v in V], 0); print('nb', s, flush=True)
out['full_044'] = np.mean([member(f'../push-800/feat-{v}H2Z-anghcor-train.npz', f'../push-800/feat-{v}H2Z-anghcor-test.npz') for v in V], 0); print('full 044', flush=True)
np.savez(_repo_io(f'fin-nb-tab-{dev}.npz'), ids=np.array(order), **out)
