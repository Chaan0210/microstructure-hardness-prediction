from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
v='u1dmavgH2Z'; F=np.load(_repo_io(f'experiments/push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys=[str(k) for k in F['keys']]; ni=keys.index('noise')
X=F['X'].astype(float); nz=np.nan_to_num(X[:,ni]*255,nan=0); R=X[nz>=12]
S=np.vstack([ (lambda S: S['X'].astype(float)[S['X'][:,ni].astype(float)*255>=12])(np.load(_repo_io(f'experiments/push-1200/synthrows-{s}-{v}-anghcor.npz'),allow_pickle=True)) for s in ['deg','deg2','deg3','deg4']])
imp=SimpleImputer(strategy='median').fit(np.vstack([R,S])); R=imp.transform(R); S=imp.transform(S)
def coral(S, R, full=True):
    mu_s, mu_r = S.mean(0), R.mean(0); sd = S.std(0) + 1e-9
    if not full: return (S - mu_s) / sd * R.std(0) + mu_r
    Cs = np.cov(S, rowvar=False) + 1e-3 * np.diag(np.diag(np.cov(S, rowvar=False)) + 1e-9); Cr = np.cov(R, rowvar=False) + 1e-3 * np.diag(np.diag(np.cov(R, rowvar=False)) + 1e-9)
    def msqrt(C, inv=False):
        w, V = np.linalg.eigh(C); w = np.clip(w, 1e-12, None); return (V * (w ** (-0.5 if inv else 0.5))) @ V.T
    return (S - mu_s) @ msqrt(Cs, True) @ msqrt(Cr) + mu_r
def auc(S2):
    Z=np.vstack([R,S2]); t=np.r_[np.ones(len(R)),np.zeros(len(S2))]; cv=StratifiedKFold(5,shuffle=True,random_state=0)
    Zs=(Z-Z.mean(0))/(Z.std(0)+1e-9)
    a1=roc_auc_score(t,cross_val_predict(LogisticRegression(C=0.1,max_iter=2000),Zs,t,cv=cv,method='predict_proba')[:,1])
    a2=roc_auc_score(t,cross_val_predict(HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05),Z,t,cv=cv,method='predict_proba')[:,1])
    return round(a1,3), round(a2,3)
print('real noisy', len(R), 'copies', len(S))
print('raw', auc(S)); print('mean shift', auc(S - S.mean(0) + R.mean(0))); print('diag scale', auc(coral(S,R,False))); print('CORAL full', auc(coral(S,R,True)))
