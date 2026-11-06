"""Test predictions of the plain linear partners with and without the het+intensity groups, fit on the 500 train rows (production procedure of
final_wls.py WEIGHTED=0 / push-900 gmm_partner.py final). 'full' must reproduce fin-none (ridge, logridge) and gmm-partner-test exactly."""
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
def mk(kind): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True) if kind == 'ridge' else SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0) if kind == 'log' else RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
def load(v):
    Tr = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); Te = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-test.npz'), allow_pickle=True); keys = [str(k) for k in Tr['keys']]
    assert [str(i) for i in Te['ids']] == order; return keys, [str(i) for i in Tr['ids']], Tr['X'].astype(float), np.array([lab[str(i)] for i in Tr['ids']]), Te['X'].astype(float)
out = {}
for tag in ('full', 'nb'):
    keys, tid, X, y, Xe = load('u1dmavgH2Z'); cols = [j for j, k in enumerate(keys) if tag == 'full' or k not in drop]
    out[f'ridge_{tag}'] = mk('ridge').fit(X[:, cols], y).predict(Xe[:, cols]); ls = []
    for v in V:
        keys, tid, X, y, Xe = load(v); cols = [j for j, k in enumerate(keys) if tag == 'full' or k not in drop]; ls.append(np.log(np.exp(mk('log').fit(X[:, cols], np.log(y)).predict(Xe[:, cols]))))
    out[f'logridge_{tag}'] = np.exp(np.mean(ls, 0))
src = open(_repo_io('../push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]; gp = {'__file__': '../push-900/gmm_partner.py', '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), gp)
keys, tid, X, y, Xe = load('u1dmavgH2Z'); S9 = gp['S9']
Xg = gp['gmm_feats'](np.load(_repo_io(S9 / 'object-bank-train.npz'), allow_pickle=True), tid); Xge = gp['gmm_feats'](np.load(_repo_io(S9 / 'object-bank-test.npz'), allow_pickle=True), order)
for tag in ('full', 'nb'):
    cols = [j for j, k in enumerate(keys) if tag == 'full' or k not in drop]; out[f'gmm_{tag}'] = gp['fit_predict'](np.c_[X[:, cols], Xg], y, np.c_[Xe[:, cols], Xge], 19001)[0]
N0 = np.load(_repo_io('../push-1200/fin-none.npz')); Gm = np.load(_repo_io('../push-900/gmm-partner-test.npz'))
print('reproduction max|diff| ridge', np.abs(out['ridge_full'] - N0['ridge']).max(), 'logridge', np.abs(out['logridge_full'] - N0['logridge']).max(), 'gmm', np.abs(out['gmm_full'] - Gm['pred']).max())
np.savez(_repo_io('fin-nb-partners.npz'), ids=np.array(order), **out)
