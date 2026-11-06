"""Teacher-target degradation transfer (TT-DTA), official dev (Study900 400 rows x 3 partitions x 5 folds).
Teacher: TabPFN-3.5 on the outer-training real rows; inner 5-fold cross-fit gives out-of-fold teacher predictions for every outer-training
row; rows outside dev-400 (holdout 100) get the mean of the 5 inner models. No validation-fold label enters any teacher/student fit:
copies whose source is in the validation fold are dropped. Students are trained on degraded copies (measured noise >= NOISEMIN) whose target is
the teacher prediction of their clean source (instead of the source label), optionally plus real outer-training rows with their labels.
usage: tt_dev.py VARIANT SETS(comma) OUTTAG     env: NOISEMIN (12), STUDENTS (ridge,tab,tabmix), TSEEDS (0)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, csv, time, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, _repo_io('../push-300')); import harness
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
P = Path(_repo_io('../..')); H12 = Path(_repo_io('../push-1200'))
v, sets, tag = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
NMIN = float(os.environ.get('NOISEMIN', '12')); STU = os.environ.get('STUDENTS', 'ridge,tab,tabmix').split(','); TSEEDS = [int(s) for s in os.environ.get('TSEEDS', '0').split(',')]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise')
ids_all = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(ids_all)}; XA = F['X'].astype(float); yA = np.array([lab[i] for i in ids_all])
SX, SID = [], []
for s in sets:
    S = np.load(_repo_path(H12, f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); assert [str(k) for k in S['keys']] == keys
    m = S['X'][:, ni].astype(float) * 255 >= NMIN; SX.append(S['X'].astype(float)[m]); SID += [str(i) for i in np.array(S['ids'])[m]]
SX = np.vstack(SX); SID = np.array(SID); print('copies', len(SID), 'sources', len(set(SID)), flush=True)
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'),
          '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
sid = load(3)['ids']
def tabpfn(seed): return harness.tabpfn_new('v3.5', seed, 'mps')
def tfit(Xt, yt, Xq):
    return np.mean([tabpfn(s).fit(Xt, yt).predict(Xq) for s in TSEEDS], 0)
cache = Path(f'teacher-{v}.npz'); TC = dict(np.load(_repo_io(cache))) if cache.exists() else {}
out = {s: np.zeros((3, 400)) for s in STU}; t0 = time.time()
for p_, (name, path) in enumerate(splits.items()):
    d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']; dev = set(ids); hold = [i for i in ids_all if i not in dev]
    for k in range(5):
        tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vset = set(va)
        key = f'{name}_{k}'
        if key + '_ids' not in TC:   # teacher cross-fit on outer-training rows
            Xtr = XA[[pos[i] for i in tr]]; ytr = yA[[pos[i] for i in tr]]; Xh = XA[[pos[i] for i in hold]]
            inner = np.random.RandomState(1000 + k).permutation(len(tr)) % 5; tp = np.zeros(len(tr)); th = []
            for j in range(5):
                a, b = inner != j, inner == j; pr = tfit(Xtr[a], ytr[a], np.vstack([Xtr[b], Xh])); tp[b] = pr[:b.sum()]; th.append(pr[b.sum():])
            TC[key + '_ids'] = np.array(tr + hold); TC[key + '_t'] = np.r_[tp, np.mean(th, 0)]; np.savez(_repo_io(cache), **TC)
            print(name, k, 'teacher done', round(time.time() - t0), 's; inner OOF rmse', round(float(np.sqrt(((tp - ytr) ** 2).mean())), 3), flush=True)
        tmap = dict(zip([str(i) for i in TC[key + '_ids']], TC[key + '_t']))
        keep = np.array([s not in vset for s in SID]); Xc = SX[keep]; tc = np.array([tmap[s] for s in SID[keep]]); yc = np.array([lab[s] for s in SID[keep]])
        Xtr = XA[[pos[i] for i in tr]]; ytr = yA[[pos[i] for i in tr]]; Xva = XA[[pos[i] for i in va]]; jj = [sid.index(i) for i in va]
        for s in STU:
            if s == 'ridge':      # RidgeCV production pipeline, real rows (labels) + copies (teacher targets)
                m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(np.vstack([Xtr, Xc]), np.r_[ytr, tc]); pr = m.predict(Xva)
            elif s == 'logr':     # log-target Ridge alpha 2, real rows + copies (teacher targets)
                m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(np.vstack([Xtr, Xc]), np.log(np.r_[ytr, tc])); pr = np.exp(m.predict(Xva))
            elif s == 'tab':      # TabPFN context = copies with teacher targets only
                pr = tfit(Xc, tc, Xva)
            elif s == 'tabmix':   # TabPFN context = real outer-training rows (labels) + copies (teacher targets)
                pr = tfit(np.vstack([Xtr, Xc]), np.r_[ytr, tc], Xva)
            elif s == 'tablab':   # control: TabPFN context = copies with their source LABELS
                pr = tfit(Xc, yc, Xva)
            out[s][p_, jj] = pr
        print(name, k, 'students done', round(time.time() - t0), 's', flush=True)
np.savez(_repo_io(f'tt-{tag}.npz'), ids=np.array(sid), **out); print('saved', f'tt-{tag}.npz')
