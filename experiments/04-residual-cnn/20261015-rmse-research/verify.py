"""Independent metric, source exclusion, held-out-label, and artifact checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv
import json
import platform
import sys
import numpy as np
import scipy
import sklearn
import joblib
from threadpoolctl import threadpool_limits
import research as r
from followups import NuisanceRidge


def main():
    d=dict(np.load(_repo_io(r.HERE/'inputs.npz'))); dev=d['dev']; fold=d['fold']; y=d['y'][dev]
    report={'environment':dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                              sklearn=sklearn.__version__,platform=platform.platform()),'checks':{}}
    r.self_check()
    saved=json.loads((r.HERE/'provenance.json').read_text())
    protected={p:sha for p,sha in saved.items() if not p.startswith(str(r.HERE.relative_to(r.ROOT))+'/')}
    changed=[p for p,sha in protected.items() if r.digest(_repo_path(r.ROOT, p))!=sha]
    assert not changed,changed
    report['checks']['protected_inputs_unchanged']=len(protected)
    report['checks']['protected_files']=protected
    records=[]
    for filename,keys in [('domain',['production','nested_control','candidate']),
                          ('consistency',['production','nested_control','candidate']),
                          ('consistency-strict',['production','nested_control','candidate']),
                          ('nuisance',['control','candidate']),
                          ('bagging',['full','penalty_control','bagged'])]:
        z=np.load(_repo_io(r.HERE/(filename+'.npz'))); info=json.loads((r.HERE/(filename+'.json')).read_text())
        for key in keys:
            pred=z[key+'_blend']; assert pred.shape==(3,400) and np.isfinite(pred).all()
            assert np.array_equal(pred[d['fail']],d['c73'][d['fail']])
            # Independent scalar accumulation, not research.metrics.
            squared=[float(pred[s,i]-y[i])**2 for s in range(3) for i in range(400)]
            pooled=(sum(squared)/len(squared))**.5
            assert abs(pooled-info[key]['candidate']['pooled'])<1e-12
            for s in range(3):
                for k in range(5):
                    idx=np.where(fold[s]==k)[0]
                    value=(sum(float(pred[s,i]-y[i])**2 for i in idx)/len(idx))**.5
                    assert abs(value-info[key]['candidate']['folds'][s][k])<1e-12
            records.append(dict(experiment=filename,arm=key,pooled=pooled))
    report['checks']['independently_recomputed_metrics']=records
    report['checks']['label_poisoning']={}
    train,valid=dev[fold[0]!=0],dev[fold[0]==0]
    poisoned={k:v.copy() for k,v in d.items()}; poisoned['y'][valid]+=10000
    for family in ('domain','consistency'):
        models=[]
        for dd in (d,poisoned):
            x,yt,dom,source,isreal,pairs,cols=r.training(dd,train,valid,True,True)
            (control,m),selection=r.fit_nested(x,yt,dom,source,isreal,pairs,True,family)
            models.append((m,selection))
        assert models[0][1]==models[1][1]
        assert np.array_equal(models[0][0].coef,models[1][0].coef)
        query=d['X'][valid][:,cols]; qdom=(d['noise'][valid]>=12).astype(int)
        p=models[0][0].predict(query,qdom)
        assert np.array_equal(p,models[1][0].predict(query,qdom))
        reverse=np.arange(len(valid))[::-1]
        assert np.allclose(p,models[0][0].predict(query[reverse],qdom[reverse])[reverse],atol=1e-12)
        path=r.HERE/f'verified-{family}-fold0.joblib'; joblib.dump(models[0][0],_repo_io(path))
        assert np.array_equal(p,joblib.load(_repo_io(path)).predict(query,qdom))
        report['checks']['label_poisoning'][family]='exact selected parameters, coefficients and query predictions after held-out labels +10000; exact model reload'
    # Audit stored split-scope counts directly.
    for s in range(3):
        for k in range(5):
            tr,va=dev[fold[s]!=k],dev[fold[s]==k]
            args=r.training(d,tr,va,True,False)
            assert not set(args[3])&set(va)
            args=r.training(d,tr,va,True,True)
            assert set(args[3])<=set(tr)
    report['checks']['all_15_outer_source_exclusions']=True
    for mode in ('reproduction','retrained-linear'):
        receipt=json.loads((r.HERE/mode/'manifest.json').read_text())
        for cid in ('061','073'):
            p=r.HERE/mode/f'submission-{cid}.csv'; rows=list(csv.DictReader(p.open()))
            assert list(rows[0])==['ID','hardness'] and len(rows)==1000
            assert r.digest(p)==receipt['outputs'][cid]['sha256']
        report['checks'][mode]=receipt['outputs']
    report['limitations']=[
        'New heads are label-safe inside outer and inner folds; inherited image representations and historical model/feature choices were not retrained from scratch inside folds.',
        'Repeated 400-ID OOF is conditional validation, not 1200 independent observations. Bootstrap intervals do not correct prior adaptive selection.',
        'Raw component differences at segmentation-failure rows are irrelevant after the preserved p021 fallback; blended production controls reproduce the baseline.',
        'No private/test labels, leaderboard queries, new external models, GPU training, or upload in this study.'
    ]
    r.save_json(r.HERE/'verification.json',report)
    print(json.dumps({'protected_inputs_unchanged':len(protected),'metric_tables_verified':len(records),'label_poisoning':'passed','reproduction':'061 and 073 byte-identical'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
