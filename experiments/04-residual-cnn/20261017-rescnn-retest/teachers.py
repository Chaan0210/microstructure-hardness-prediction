"""Nested T teachers for the re-test (PLAN.md). Imports round4's full_branch read-only (run python with -B so nothing is written there) and
redirects only its teacher output directory to this experiment. Recipe unchanged.
Jobs: partitions p=1 (20261031), p=2 (20261101): each outer fold k -> outer0 (A->B), outer1 (T-A1 -> A1), outer2 (T-A2 -> A2).
Holdout (p=3): T = all 400 dev rows, V = the 100 holdout rows -> outer0/1/2 as above plus full (T -> V).
usage: python -B teachers.py graph            -> label-graph.json
       python -B teachers.py run WORKER NWORK -> teachers/... (jobs split round-robin across workers)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os, sys, json
from pathlib import Path
OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(_repo_path(OUT, 'mpl-cache')); os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1'); sys.dont_write_bytecode = True
R4 = _repo_path(OUT.parent, 'rmse-research-20261015/round4-full073')
sys.path.insert(0, str(R4))
import numpy as np

def graph():
    d = dict(np.load(_repo_path(R4.parent, 'inputs.npz'))); dev = d['dev']; allix = np.arange(500); hold = np.setdiff1d(allix, dev)
    G = []
    def nodes(T, V, seed):
        rg = np.random.RandomState(seed); perm = rg.permutation(T); nB = int(round(len(T) / 3)); B = np.sort(perm[:nB]); A = perm[nB:]
        pa = rg.permutation(A); h = int(np.ceil(len(A) / 2)); A1 = np.sort(pa[:h]); A2 = np.sort(pa[h:])
        return dict(T=np.sort(T).tolist(), V=np.sort(V).tolist(), A=np.sort(A).tolist(), B=B.tolist(), A1=A1.tolist(), A2=A2.tolist(),
                    nodes=[dict(name='outer0', train=np.sort(A).tolist(), query=B.tolist()),
                           dict(name='outer1', train=np.setdiff1d(T, A1).tolist(), query=A1.tolist()),
                           dict(name='outer2', train=np.setdiff1d(T, A2).tolist(), query=A2.tolist())])
    for p in (1, 2):
        for k in range(5):
            V = dev[d['fold'][p] == k]; T = dev[d['fold'][p] != k]; g = nodes(T, V, 20261017 + 100 * p + k); g.update(partition=p, fold=k); G.append(g)
    g = nodes(dev, hold, 20261017 + 300); g['nodes'].append(dict(name='full', train=np.sort(dev).tolist(), query=np.sort(hold).tolist())); g.update(partition=3, fold=0); G.append(g)
    for g in G:   # integrity: every node's train/query disjoint; outer teachers cover T exactly once; nothing touches V
        cov = np.zeros(500, int); V = set(g['V'])
        for n in g['nodes']:
            assert not set(n['train']) & set(n['query'])
            if n['name'] != 'full': assert not (set(n['train']) | set(n['query'])) & V; cov[n['query']] += 1
        assert np.all(cov[g['T']] == 1) and cov.sum() == len(g['T'])
    json.dump(G, open(_repo_path(OUT, 'label-graph.json'), 'x'), indent=0); print('graph', len(G), 'folds,', sum(len(g['nodes']) for g in G), 'teachers')

def run(worker, nwork):
    import full_branch as fb
    d, labels = fb.load()                      # reads inputs/feature banks/copies from their original read-only locations
    if os.environ.get('TAB_DEVICE'):          # amendment A1: execution device only (CPU under 3-way contention ~17 s/fit); same model, same seeds
        _orig = fb.harness.tabpfn_new; fb.harness.tabpfn_new = lambda v, s, dev, n_est=None: _orig(v, s, os.environ['TAB_DEVICE'], n_est)
    fb.HERE = OUT                              # from here on fit() writes teachers under OUT/teachers/ only
    G = json.load(open(_repo_path(OUT, 'label-graph.json')))
    jobs = [(g['partition'], g['fold'], n) for g in G for n in g['nodes']]
    for j, (p, k, n) in enumerate(jobs):
        if j % nwork != worker: continue
        train = np.array(n['train'], int); query = np.array(n['query'], int)
        fb.fit(d, train, query, labels[train].copy(), k, f"p{p}-{n['name']}")   # seeds use the fold index k, as in round4
        print(json.dumps(dict(done=f"p{p} fold{k} {n['name']}", worker=worker)), flush=True)

if __name__ == '__main__':
    graph() if sys.argv[1] == 'graph' else run(int(sys.argv[2]), int(sys.argv[3]))
