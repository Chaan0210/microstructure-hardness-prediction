"""Official dev procedure for the noisy-row model trained on [real noisy outer-train rows + synthetic degraded copies of clean outer-train images].
Base = saved dev OOF of an existing tag (grain prediction for all rows). Noisy val rows get MODE: replace | avg (with base).
usage: dev_synth.py BASE_TAG(v2 or v35 dev npz tag) REAL_FEAT_NPZ SYNTH_FEAT_NPZ SYNTH_UNET_NPZ MODEL(v2|v3.5) OUT_TAG [device=cpu]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import harness
btag, rf, sf, su, model, out = sys.argv[1:7]; device = sys.argv[7] if len(sys.argv) > 7 else 'cpu'
B = np.load(_repo_io(f'dev-{btag}.npz'), allow_pickle=True)
R = np.load(_repo_io(rf), allow_pickle=True); keys = [str(k) for k in R['keys']]; rid = [str(i) for i in R['ids']]; XR = R['X'].astype(float); rpos = {i: k for k, i in enumerate(rid)}
A = np.load(_repo_io(sf), allow_pickle=True); ak = [str(k) for k in A['keys']]; U = np.load(_repo_io(su), allow_pickle=True); uk = [str(k) for k in U['keys']]; sid = [str(i) for i in A['ids']]
XS = np.stack([A['X'][:, ak.index(k)] if k in ak else U['X'][:, uk.index(k)] for k in keys], 1); spos = {i: k for k, i in enumerate(sid)}
nzall = np.load(_repo_io('noise_train.npy')); allids = [str(i) for i in np.load(_repo_io('../grain-cellpose-220/features-v3-train.npz'), allow_pickle=True)['ids']]
med = np.median(nzall); nzd = dict(zip(allids, nzall))
lab = harness.labels(allids); yd = dict(zip(allids, lab))
def make(sd, k):
    return harness.tabpfn(32, 100 * sd + k, device) if model == 'v2' else harness.tabpfn_new(model, 100 * sd + k, device)
save = {}; res = {}
for name in ['original', '20261031', '20261101']:
    ids = [str(i) for i in B[f'{name}_ids']]; y = B[f'{name}_y']; fold = B[f'{name}_fold']; g = B[f'{name}_g'].copy()
    noisy = np.array([nzd[i] >= med for i in ids]); gs = g.copy()
    for k in range(5):
        tr, va = fold != k, (fold == k) & noisy
        trn_ids = [i for i, t in zip(ids, tr) if t]
        real = [i for i in trn_ids if nzd[i] >= med]; syn = [i for i in trn_ids if i in spos]
        Xt = np.vstack([XR[[rpos[i] for i in real]], XS[[spos[i] for i in syn]]]); yt = np.r_[[yd[i] for i in real], [yd[i] for i in syn]]
        Xv = XR[[rpos[i] for i, v in zip(ids, va) if v]]
        ps = []
        for sd in (0, 1, 2):
            m = make(sd, k); m.fit(Xt, yt); ps.append(m.predict(Xv))
        gs[va] = np.mean(ps, 0)
    rm = lambda v, m=None: float(np.sqrt(np.mean(((v - y) if m is None else (v - y)[m]) ** 2)))
    res[name] = dict(base_grain=rm(g), synth_replace=rm(gs), synth_avg=rm((g + gs) / 2), base_noisy=rm(g, noisy), synth_noisy=rm(gs, noisy), avg_noisy=rm((g + gs) / 2, noisy))
    print(name, json.dumps({k: round(v, 4) for k, v in res[name].items()}), flush=True)
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_fold': fold, f'{name}_p021': B[f'{name}_p021'], f'{name}_failed': B[f'{name}_failed'],
                 f'{name}_g': gs, f'{name}_gavg': (g + gs) / 2})
np.savez(_repo_io(f'dev-{out}.npz'), **save); json.dump(res, open(_repo_io(f'dev-{out}.json'), 'w'), indent=1)
