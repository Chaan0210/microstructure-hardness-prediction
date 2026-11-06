"""Candidate from several (feature set, model) components: grain = equal mean over components, each = mean of 3 seeds fit on all 500 train rows;
final = (1-W)*best021 + W*grain (best021 where the PRIMARY test feature file has n_obj < 20).
usage: finalize_multi.py CID W EVIDENCE DEVICE COMP [COMP ...]   COMP = model@train_npz@test_npz[@synth_feat_npz@synth_unet_npz]
  model: v2 (n_est 32) | v3.5 (default). With synth files, the training set gets the synthetic rows (flag column 1; real rows 0)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, csv, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, '.'); sys.path.insert(0, _repo_io('../grain-cellpose-220'))
import harness
from finalize_common import write_candidate, sha
cid, W, evidence, device = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]; comps = sys.argv[5:]
P = Path(_repo_io('../..')); lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
preds = []; meta = []; prim = None
for c in comps:
    parts = c.split('@'); model, ftr, fte = parts[:3]
    Tr = np.load(_repo_io(ftr), allow_pickle=True); Te = np.load(_repo_io(fte), allow_pickle=True); keys = [str(k) for k in Tr['keys']]; assert keys == [str(k) for k in Te['keys']]
    Xtr, ytr, Xte = Tr['X'].astype(float), np.array([lab[str(i)] for i in Tr['ids']]), Te['X'].astype(float)
    if len(parts) > 3:
        A = np.load(_repo_io(parts[3]), allow_pickle=True); ak = [str(k) for k in A['keys']]; U = np.load(_repo_io(parts[4]), allow_pickle=True); uk = [str(k) for k in U['keys']]
        XS = np.stack([A['X'][:, ak.index(k)] if k in ak else U['X'][:, uk.index(k)] for k in keys], 1)
        Xtr = np.vstack([np.c_[Xtr, np.zeros(len(Xtr))], np.c_[XS, np.ones(len(XS))]]); ytr = np.r_[ytr, [lab[str(i)] for i in A['ids']]]; Xte = np.c_[Xte, np.zeros(len(Xte))]
    ps = []
    for sd in (0, 1, 2):
        est = harness.tabpfn(32, 100 * sd, device) if model == 'v2' else harness.tabpfn_new(model, 100 * sd, device)
        est.fit(Xtr, ytr); ps.append(est.predict(Xte))
    preds.append(np.mean(ps, 0)); meta.append(dict(component=c, train_sha256=sha(ftr), test_sha256=sha(fte), n_train_rows=int(len(ytr))))
    if prim is None: prim = (Te, keys)
    print(c, 'seed corr', [round(float(np.corrcoef(ps[0], ps[k])[0, 1]), 5) for k in (1, 2)], flush=True)
g = np.mean(preds, 0); Te, keys = prim
write_candidate(cid, g, [str(i) for i in Te['ids']], keys, Te['X'], W, 20,
                dict(formula=f'{1 - W:.2f}*best021 + {W:.2f}*mean over components (fallback best021 if n_obj<20)', components=meta,
                     component_corr=[[float(np.corrcoef(a, b)[0, 1]) for b in preds] for a in preds], dev_evidence=evidence, device=device))
