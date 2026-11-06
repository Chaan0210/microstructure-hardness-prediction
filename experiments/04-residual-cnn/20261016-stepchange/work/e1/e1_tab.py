"""E1 (see PLAN.md): noisy-row TabPFN with/without degraded copies on all columns vs the label-free domain-invariant subset.
usage: e1_tab.py ARM [device]   ARM in F0 F1 A0 A1 A2 C1  -> work/e1/oof-ARM.npz (OOF on all 400 dev rows, 3 partitions)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, json, time, numpy as np, torch, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[4]; ROOT = _RepoPath(__file__).resolve().parents[5]
sys.path.insert(0, str(_repo_path(E, 'push-300'))); cwd = os.getcwd(); os.chdir(_repo_path(E, 'push-300')); import harness; os.chdir(_repo_io(cwd))
arm = sys.argv[1]; device = sys.argv[2] if len(sys.argv) > 2 else 'mps'; torch.set_num_threads(2)
out = _repo_path(HERE, f'oof-{arm}.npz'); assert not out.exists()
v = 'u1dmavgH2Z'
F = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; XA = F['X'].astype(float); pos = {str(i): k for k, i in enumerate(F['ids'])}
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(ROOT, 'data/train.csv')))}
ni = keys.index('noise')
inv = json.load(open(_repo_path(HERE, f'domain-{v}.json')))['multi_dev<=0.05']['cols']
cols = list(range(len(keys))) if arm.startswith('F') else sorted(set([keys.index(k) for k in inv] + [ni]))
SX, SID, SSET = [], [], []
for s in (['deg4'] if arm == 'A2' else ['deg', 'deg2', 'deg3', 'deg4']):
    S = np.load(_repo_path(E, f'push-1200/synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); assert [str(k) for k in S['keys']] == keys
    m = S['X'][:, ni].astype(float) * 255 >= 12; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
SX = np.vstack(SX); SID = np.array(SID)
use_copies = arm in ('F1', 'A1', 'A2', 'C1')
srcs = sorted(set(SID)); perm = dict(zip(srcs, np.random.RandomState(20261016).permutation(srcs)))   # C1: copy labels permuted across sources
splits = {'original': _repo_path(E, 'resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(E, 'resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(E, 'resolution-repartition-217/20261101/ensemble-OOF.npz')}
save = {}; t0 = time.time()
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]
    g = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k; Xt, yt = X[tr][:, cols], y[tr]
        if use_copies:
            trs = set(np.array(ids)[tr]); keep = np.array([s in trs for s in SID])
            yc = np.array([lab[perm[s]] if arm == 'C1' else lab[s] for s in SID[keep]])
            Xt = np.vstack([Xt, SX[keep][:, cols]]); yt = np.r_[yt, yc]
        ps = []
        for sd in (0, 1, 2):
            m = harness.tabpfn_new('v3.5', 100 * sd + k, device); m.fit(Xt, yt); ps.append(m.predict(X[va][:, cols]))
        g[va] = np.mean(ps, 0)
        print(arm, name, k, len(yt), f'{time.time() - t0:.0f}s', flush=True)
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_fold': fold, f'{name}_g': g})
np.savez(_repo_io(out), cols=np.array([keys[j] for j in cols]), arm=arm, **save); print('saved', out)
