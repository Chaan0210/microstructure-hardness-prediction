"""[clip: the 2 H9 columns clipped to the train range before use] Final (500 train rows) plain partners with nb columns + 2 H9 cumulant columns (test cumulants from cumul-test.npz). Same procedure as final_nb_partners.py
(which reproduced production exactly). -> fin-h9c-partners.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, json, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}; order = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/sample_submission.csv')))]
G = json.load(open(_repo_io('abl/groups.json'))); drop = set(G['intensity'].split() + G['het'].split()); V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
C = {sp: np.load(_repo_io(f'cumul-{sp}.npz')) for sp in ('train', 'test')}; cp = {sp: {str(i): j for j, i in enumerate(C[sp]['ids'])} for sp in C}
def mk(kind): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True) if kind == 'ridge' else SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
def load(v):
    Tr = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); Te = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-test.npz'), allow_pickle=True); keys = [str(k) for k in Tr['keys']]
    tid = [str(i) for i in Tr['ids']]; assert [str(i) for i in Te['ids']] == order; keep = [j for j, k in enumerate(keys) if k not in drop]
    ex = lambda sp, ids: np.clip(C[sp]['X'][[cp[sp][i] for i in ids]][:, [1, 4]], C['train']['X'][:, [1, 4]].min(0), C['train']['X'][:, [1, 4]].max(0))
    return tid, np.c_[Tr['X'].astype(float)[:, keep], ex('train', tid)], np.array([lab[i] for i in tid]), np.c_[Te['X'].astype(float)[:, keep], ex('test', order)]
out = {}
tid, X, y, Xe = load('u1dmavgH2Z'); out['ridge'] = mk('ridge').fit(X, y).predict(Xe)
out['logridge'] = np.exp(np.mean([mk('log').fit(load(v)[1], np.log(load(v)[2])).predict(load(v)[3]) for v in V], 0))
src = open(_repo_io('../push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]; gp = {'__file__': '../push-900/gmm_partner.py', '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), gp)
S9 = gp['S9']; Xg = gp['gmm_feats'](np.load(_repo_io(S9 / 'object-bank-train.npz'), allow_pickle=True), tid); Xge = gp['gmm_feats'](np.load(_repo_io(S9 / 'object-bank-test.npz'), allow_pickle=True), order)
F = load('u1dmavgH2Z'); out['gmm'] = gp['fit_predict'](np.c_[F[1][:, :-2], Xg, F[1][:, -2:]], F[2], np.c_[F[3][:, :-2], Xge, F[3][:, -2:]], 19001)[0]
np.savez(_repo_io('fin-h9c-partners.npz'), ids=np.array(order), **out); print('saved')
