"""Build a candidate from fine-tuned-segmenter features: (1-W)*best021 + W*grain, grain = mean over models of TabPFN fits on all 500 train rows.
usage: finalize_ft.py CID TRAIN_NPZ TEST_NPZ W MODELS EVIDENCE [device=cpu]
  MODELS: comma list of v2 (n_est 32 x seeds 0,1,2) and/or v3.5 (default n_est x seeds 0,1,2)
  EVIDENCE: free text pointing at the dev result that justified this candidate (recorded in the summary)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, csv, hashlib, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, _repo_io('../grain-cellpose-220'))
import harness
from finalize_common import write_candidate, sha
cid, ftr, fte, W, models, evidence = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5].split(','), sys.argv[6]
device = sys.argv[7] if len(sys.argv) > 7 else 'cpu'
P = Path(_repo_io('../..'))
Ftr = np.load(_repo_io(ftr), allow_pickle=True); Fte = np.load(_repo_io(fte), allow_pickle=True); keys = [str(k) for k in Ftr['keys']]; assert keys == [str(k) for k in Fte['keys']]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
Xtr, ytr, Xte = Ftr['X'], np.array([lab[str(i)] for i in Ftr['ids']]), Fte['X']
per = {}
for m in models:
    ps = []
    for sd in (0, 1, 2):
        est = harness.tabpfn(32, 100 * sd, device) if m == 'v2' else harness.tabpfn_new(m, 100 * sd, device)
        est.fit(Xtr, ytr); ps.append(est.predict(Xte))
    per[m] = np.mean(ps, 0); print(m, 'seed corr', [round(float(np.corrcoef(ps[0], ps[k])[0, 1]), 5) for k in (1, 2)], flush=True)
g = np.mean([per[m] for m in models], 0)
write_candidate(cid, g, [str(i) for i in Fte['ids']], keys, Xte, W, 20,
                dict(formula=f'{1 - W:.2f}*best021 + {W:.2f}*mean({"+".join(models)}) TabPFN on {Path(ftr).name}/{Path(fte).name} (fallback best021 if n_obj<20)',
                     train_features=str(Path(ftr).resolve().relative_to(P.resolve())), train_features_sha256=sha(ftr), test_features=str(Path(fte).resolve().relative_to(P.resolve())), test_features_sha256=sha(fte),
                     models=models, model_corr={f'{a}~{b}': float(np.corrcoef(per[a], per[b])[0, 1]) for a in per for b in per if a < b}, dev_evidence=evidence))
