"""Bare-X98 control with exactly stage2's inner folds and Ridge selection.

This isolates added measurements from the change of GCV to threefold selection.
No additional hyperparameters or post-hoc blend weights are searched.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from common import HERE,load_base,evaluate,destination,dump_json


def main():
    b=load_base();out=destination(HERE/'matched-controls/placeholder').parent
    pred=np.full_like(b['baseline'],np.nan);choices=[]
    alphas=[.1,1.,10.,100.,1000.]
    for s in range(3):
        for f in range(5):
            tr=np.flatnonzero(b['fold'][s]!=f);va=np.flatnonzero(b['fold'][s]==f)
            inner=[]
            for k in range(3):
                c=np.load(_repo_io(HERE/f'probabilistic/contrast-adaptive/stage2/features/partition{s}-fold{f}-inner{k}.npz'))
                a,z=c['train_idx'],c['eval_idx'];assert set(a)|set(z)==set(tr);assert not set(a)&set(z);inner.append((a,z))
            scores=[]
            for alpha in alphas:
                errors=[]
                for a,z in inner:
                    m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha))
                    m.fit(b['X98'][a],b['y'][a]);errors.extend((m.predict(b['X98'][z])-b['y'][z])**2)
                scores.append(float(np.mean(errors)))
            alpha=alphas[int(np.argmin(scores))]
            m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha))
            m.fit(b['X98'][tr],b['y'][tr]);pred[s,va]=m.predict(b['X98'][va])
            choices.append(dict(partition=s,fold=f,alpha=alpha,inner_mse=scores))
    p=np.where(b['failed'],b['p021'],pred)
    r={'standalone':evaluate(p,b),'blend20':evaluate(.8*b['baseline']+.2*p,b),
       'same021':evaluate(.2*b['p021']+.8*(.8*b['grain']+.2*p),b),'selection':choices,
       'purpose':'Matched post-study attribution control, not a newly selected deployment recipe.'}
    dump_json(out/'results.json',r);np.savez_compressed(_repo_io(out/'predictions.npz'),ids=b['ids'],prediction=pred)
    print({k:r[k]['rmse'] for k in ['standalone','blend20','same021']})


if __name__=='__main__':main()
