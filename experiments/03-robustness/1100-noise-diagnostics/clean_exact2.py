"""clean_exact.py follow-up: random-column nulls (13 N(0,1) columns, 3 draws) and group ablations of the 13 exact columns, same folds/seeds."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; sys.path.insert(0, str(_repo_path(E, 'push-300'))); import harness
from sklearn.model_selection import KFold
G = np.load(_repo_path(E, 'push-1000/geom-exact-train.npz'), allow_pickle=True); ids = [str(i) for i in G['ids']]; XG = G['X'].astype(float); KG = [str(k) for k in G['keys']]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(E.parent, 'data/train.csv')))}; y = np.array([lab[i] for i in ids])
F = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); pos = {str(i): j for j, i in enumerate(F['ids'])}; X98 = F['X'][[pos[i] for i in ids]].astype(float)
grp = {'size': ['log_density', 'log_mean_ecd', 'log_aw_ecd', 'ecd_cv'], 'phase': ['phase_area_frac', 'phase_count_frac'], 'shape': ['aw_log_ar', 'aw_c2', 'aw_s2'],
       'pore': ['pore_area_frac', 'log_pore_density'], 'bnd': ['boundary_frac', 'interior_frac']}
sets = {f'rnd{s}': np.c_[X98, np.random.RandomState(900 + s).normal(size=(len(y), 13))] for s in range(3)}
sets.update({f'+{g}': np.c_[X98, XG[:, [KG.index(c) for c in cs]]] for g, cs in grp.items()})
sets['-size'] = np.c_[X98, XG[:, [KG.index(c) for g, cs in grp.items() if g != 'size' for c in cs]]]
P = {k: np.zeros((3, len(y))) for k in sets}
for r in range(3):
    for f, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=100 + r).split(y)):
        for k, X in sets.items():
            m = harness.tabpfn_new('v3.5', 10 * r + f, 'mps'); m.fit(X[tr], y[tr]); P[k][r, va] = m.predict(X[va])
    print('repeat', r, flush=True)
for k in sets: print(f'{k:8s} RMSE', np.sqrt(((P[k] - y) ** 2).mean()).round(4), ' per repeat', np.sqrt(((P[k] - y) ** 2).mean(1)).round(3), flush=True)
np.savez(_repo_path(H, 'clean_exact2.npz'), ids=np.array(ids), y=y, **P)
