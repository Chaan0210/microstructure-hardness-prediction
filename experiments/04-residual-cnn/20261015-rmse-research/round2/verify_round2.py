"""Check protected inputs, source membership, metrics and honest selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import sklearn
import scipy
import joblib
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from models import HERE,inputs,EvidenceRidge,PLSQuadratic,groups
from cumulants import ExtraRidge
import research as r


def main():
    d=inputs();report={};start=time.monotonic()
    manifest=json.loads((HERE/'run-manifest.json').read_text())
    mismatch=[p for p,sha in manifest['files'].items()if r.digest(_repo_path(r.ROOT, p))!=sha]
    assert not mismatch,mismatch
    protected=json.loads((r.HERE/'provenance.json').read_text())
    protected={p:s for p,s in protected.items()if not p.startswith(str(r.HERE.relative_to(r.ROOT))+'/')}
    assert all(r.digest(_repo_path(r.ROOT, p))==s for p,s in protected.items())
    report['source_and_manifest_hashes_match']=True;report['protected_external_files_unchanged']=len(protected)
    # Isotropic evidence posterior must equal Ridge with its inferred precision ratio.
    rng=np.random.default_rng(r.SEED);x=rng.normal(size=(100,9));y=x[:,0]+rng.normal(size=100)
    m=EvidenceRidge(np.zeros(9,int),max_iter=2000).fit(x,y)
    z=m.scaler.transform(m.imputer.transform(x))
    ref=Ridge(alpha=m.precision[0]/m.beta).fit(z,y).predict(z)
    assert np.max(np.abs(ref-m.predict(x)))<1e-10
    report['isotropic_ridge_equivalence_maxdiff']=float(np.max(np.abs(ref-m.predict(x))))
    # All newly fitted head families accept training arrays, not full OOF residuals.
    dev=d['dev'];fold=d['fold'][0];tr,va=dev[fold!=0],dev[fold==0]
    cols=d['keep'];a=d['X'][tr][:,cols];b=d['X'][va][:,cols];g,_=groups(d['keys'][cols])
    poison={k:v.copy()if isinstance(v,np.ndarray)else v for k,v in d.items()};poison['y'][va]+=10000
    models=[EvidenceRidge(g,2000),PLSQuadratic(10,.1),ExtraRidge(10,.1)]
    ex=np.load(_repo_io(HERE/'cumulants-train.npz'))['X']
    checks=[]
    for model in models:
        if isinstance(model,ExtraRidge):
            model.fit(a,ex[tr],d['y'][tr]);p=model.predict(b,ex[va]);model.fit(a,ex[tr],poison['y'][tr]);pp=model.predict(b,ex[va])
        else:
            model.fit(a,d['y'][tr]);p=model.predict(b);model.fit(a,poison['y'][tr]);pp=model.predict(b)
        assert np.array_equal(p,pp)
        path=HERE/(type(model).__name__+'-verified.joblib');joblib.dump(model,_repo_io(path));loaded=joblib.load(_repo_io(path))
        rp=loaded.predict(b,ex[va])if isinstance(model,ExtraRidge)else loaded.predict(b)
        assert np.array_equal(p,rp)
        checks.append(dict(model=type(model).__name__,poison_delta=0.,reload_delta=0.))
    report['head_poison_reload_checks']=checks
    # Every inner base in H8 has a strict subset of the current outer train IDs.
    root_sets=[];metric_rows=[]
    for seed in (2026101501,2026101502,2026101503):
        root=HERE/f'full500-{seed}'
        for k in range(5):
            z=np.load(_repo_io(root/f'fold{k}.npz'));record=json.loads((root/f'fold{k}.json').read_text())
            ti,vi=set(z['train_ids']),set(z['validation_ids']);assert len(ti)==400 and len(vi)==100 and not ti&vi
            inner_coverage=[];base=[];delta=[];ids=[]
            for j in range(3):
                q=np.load(_repo_io(root/f'fold{k}-inner{j}.npz'));it,iv=set(q['train_ids']),set(q['validation_ids'])
                assert not it&iv and it|iv==ti and not(it|iv)&vi
                inner_coverage.extend(q['validation_ids']);base.extend(q['base']);delta.extend(q['group_delta']);ids.extend(q['validation_ids'])
            assert set(inner_coverage)==ti and len(inner_coverage)==400
            lookup={str(i):j for j,i in enumerate(d['ids'])};ii=np.array([lookup[str(i)]for i in ids]);nz=d['noise'][ii]
            for band in (0,1):
                mask=((nz>=12)==bool(band))&~d['failed500'][ii] if 'failed500'in d else ((nz>=12)==bool(band))&(np.nan_to_num(d['X'][ii,list(d['keys']).index('n_obj')],nan=0)>=20)
                mse=[np.mean((np.array(base)[mask]+v*np.array(delta)[mask]-d['y'][ii[mask]])**2)for v in (0.,.25,.5,1.)]
                assert np.allclose(mse,record['inner_selection'][band]['mse'],atol=1e-12)
                assert (0.,.25,.5,1.)[int(np.argmin(mse))]==record['lambdas'][band]
            vaidx=z['validation_index'];yy=d['y'][vaidx]
            for key in ['base','nested_blend','group_all_blend','group_low_blend','cumulants_blend','random0_blend','random1_blend','random2_blend']:
                pred=z[key];value=(sum(float(a-b)**2 for a,b in zip(pred,yy))/100)**.5
                metric_rows.append([seed,k,key,value])
            assert abs(r.rmse(z['base'],yy)-record['base_rmse'])<1e-12
            root_sets.append(dict(seed=seed,fold=k,outer_train=400,outer_valid=100,inner_partition_verified=True))
    report['all_nested_source_partitions']=root_sets
    with(HERE/'fresh-fold-metrics.csv').open('w',newline='')as f:
        w=csv.writer(f);w.writerow(['seed','fold','arm','RMSE']);w.writerows(metric_rows)
    s=np.load(_repo_io(HERE/'full500-summary.npz'));j=json.loads((HERE/'full500-summary.json').read_text())
    for key in j:
        if key in s and key!='base':
            pred=s[key];assert np.isfinite(pred).all()
            direct=float(np.linalg.norm(pred-d['y'])/np.sqrt(pred.size))
            assert abs(direct-j[key]['candidate']['pooled'])<1e-12
    report['fresh_pooled_metrics_independently_verified']=True
    report['environment']=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,platform=platform.platform())
    report['seconds']=time.monotonic()-start
    r.save_json(HERE/'verification.json',report)
    print('Passed: protected hashes, all 45 inner/15 outer source partitions, selection algebra, metrics, head poisoning and reload.')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
