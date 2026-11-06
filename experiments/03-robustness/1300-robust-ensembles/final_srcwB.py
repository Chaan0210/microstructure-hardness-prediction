"""Final (500 train rows) DTA partners, arm B of srcw.py: unweighted, alpha chosen by inner 5-fold CV grouped by source (copies with their source),
scored on real rows with noise >= 12; copies deg..deg4 with measured noise >= 12 of all train sources. RidgeCV-grid Ridge (u1dmavgH2Z) and logRidge (4 variants).
-> fin-srcwB.npz (test predictions). Nothing fitted on test."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, numpy as np, warnings
warnings.filterwarnings('ignore')
src = open(_repo_io('srcw.py')).read(); exec(compile(src[:src.index('D = {v: data(v)')], 'srcw', 'exec'))
order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]; out = {}; al = {}
for kind, vs in (('ridge', ['u1dmavgH2Z']), ('log', V)):
    preds = []
    for v in vs:
        pos, XA, NZ, SX, SID = data(v); Te = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-test.npz'), allow_pickle=True); assert [str(i) for i in Te['ids']] == order
        ids = list(pos); jt = np.array([pos[i] for i in ids])
        m, a = fit_arm('B', kind, XA[jt], np.array([lab[i] for i in ids]), np.array(ids), NZ[jt], SX, np.array([lab[s] for s in SID]), SID)
        preds.append(itf(kind, m.predict(Te['X'].astype(float)))); al[f'{kind}_{v}'] = a
    out['ridge' if kind == 'ridge' else 'logridge'] = preds[0] if kind == 'ridge' else np.exp(np.mean(np.log(preds), 0))
print('alphas', al); np.savez(_repo_io('fin-srcwB.npz'), ids=np.array(order), **out)
G = np.load(_repo_io('../push-1200/fin-d1234n12.npz')); print('test gated partners change vs production: ridge rms %.3f logridge rms %.3f' % tuple(np.sqrt(((out[k] - G[k])[np.nan_to_num(G['noise'], nan=0) >= 12] ** 2).mean()) for k in ('ridge', 'logridge')))
