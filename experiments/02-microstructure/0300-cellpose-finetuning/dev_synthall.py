"""Official dev procedure: TabPFN on [real outer-train rows (flag 0) + synthetic degraded copies of clean outer-train images (flag 1)] -> predict outer-val rows.
usage: dev_synthall.py REAL_FEAT_NPZ SYNTH_FEAT_NPZ[,SYNTH2] SYNTH_UNET_NPZ[,UNET2] MODEL(v2|v3.5) OUT_TAG [device=cpu]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import harness
from pathlib import Path
P = Path(_repo_io('../..'))
rf, sfs, sus, model, out = sys.argv[1:6]; device = sys.argv[6] if len(sys.argv) > 6 else 'cpu'
R = np.load(_repo_io(rf), allow_pickle=True); keys = [str(k) for k in R['keys']]; rid = [str(i) for i in R['ids']]; XR = np.c_[R['X'].astype(float), np.zeros(len(rid))]; rpos = {i: k for k, i in enumerate(rid)}
SX, SID = [], []
for sf, su in zip(sfs.split(','), sus.split(',')):
    A = np.load(_repo_io(sf), allow_pickle=True); ak = [str(k) for k in A['keys']]; U = np.load(_repo_io(su), allow_pickle=True); uk = [str(k) for k in U['keys']]
    sid = [str(i) for i in A['ids']]; assert sid == [str(i) for i in U['ids']]
    SX.append(np.c_[np.stack([A['X'][:, ak.index(k)] if k in ak else U['X'][:, uk.index(k)] for k in keys], 1), np.ones(len(sid))]); SID += sid
XS = np.vstack(SX); SID = np.array(SID)
allids = [str(i) for i in np.load(_repo_io('../grain-cellpose-220/features-v3-train.npz'), allow_pickle=True)['ids']]; yd = dict(zip(allids, harness.labels(allids)))
o3 = np.load(_repo_io('../grain-cellpose-220/oof-v3.npz'))
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
def make(sd, k): return harness.tabpfn(32, 100 * sd + k, device) if model == 'v2' else harness.tabpfn_new(model, 100 * sd + k, device)
save = {}; res = {}; nkey = keys.index('n_obj')
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids = [str(i) for i in d['ids']]; y, fold = d['y'], d['fold']; a = f'B_v3feat_v3mask_{name}_'; p = 2 * o3[a + 'b'] - o3[a + 'g']
    X = XR[[rpos[i] for i in ids]]; failed = np.nan_to_num(X[:, nkey], nan=0) < 20; g = np.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k; trs = set(np.array(ids)[tr]); ks = np.array([i in trs for i in SID])
        Xt = np.vstack([X[tr], XS[ks]]); yt = np.r_[y[tr], [yd[i] for i in SID[ks]]]; ps = []
        for sd in (0, 1, 2):
            m = make(sd, k); m.fit(Xt, yt); ps.append(m.predict(X[va]))
        g[va] = np.mean(ps, 0)
    res[name] = dict(grain=float(np.sqrt(np.mean((g - y) ** 2))), n_synth_used=int(ks.sum()))
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_fold': fold, f'{name}_p021': p, f'{name}_g': g, f'{name}_failed': failed})
    print(name, json.dumps(res[name]), flush=True)
np.savez(_repo_io(f'dev-{out}.npz'), **save); json.dump(res, open(_repo_io(f'dev-{out}.json'), 'w'), indent=1)
