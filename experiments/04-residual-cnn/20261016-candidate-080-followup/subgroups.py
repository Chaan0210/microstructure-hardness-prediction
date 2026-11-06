"""Independent aggregation and fixed-noise decomposition of the same E3 runs."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
sys.dont_write_bytecode=True
from collections import defaultdict
import json
import numpy as np
from sklearn.model_selection import ShuffleSplit
from threadpoolctl import threadpool_limits
import replay as p
s=p.s;H=p.H


def metrics(errors,mask):
    m=errors[:,mask]
    return dict(n_unique_ids=int(mask.sum()),base_rmse=float(np.sqrt(m[0].mean())),
                aux_rmse=float(np.sqrt(m[1].mean())),context_rmse=float(np.sqrt(m[2].mean())),
                aux_gain=float(np.sqrt(m[0].mean())-np.sqrt(m[1].mean())),
                image_increment=float(np.sqrt(m[2].mean())-np.sqrt(m[1].mean())))


def main():
    d=s.load_inputs();source=json.loads((H/'resplit-results.json').read_text());data=[];parts=[];observations=[]
    for k in s.ANCHORS:
        z=dict(np.load(_repo_io(H/f'anchor{k}-heldout187.npz')));y=d['y'][z['query']]
        splits=list(ShuffleSplit(n_splits=5,train_size=107,test_size=80,random_state=s.SPLIT_SEED).split(y));data.append((k,z,y,splits))
        for j,(cal,va) in enumerate(splits):
            pa,_=p.calibrate_predict(z['base'],z['aux'],z['noise'],z['failed'],y,cal,va)
            pc,_=p.calibrate_predict(z['base'],z['context'],z['noise'],z['failed'],y,cal,va)
            err=np.c_[(z['base'][va]-y[va])**2,(pa-y[va])**2,(pc-y[va])**2]
            parts.append(dict(ids=z['ids'][va].tolist(),base=err[:,0],aux=err[:,1],context=err[:,2]))
            for idx,row in zip(va,err):observations.append((str(z['ids'][idx]),k,j,float(z['noise'][idx]),*row.tolist()))
    ids,errors,summary=p.aggregate(parts)
    # Different implementation: lists grouped by ID, then average within ID.
    grouped=defaultdict(list)
    for row in observations:grouped[row[0]].append(row[4:])
    independent=np.array([np.mean(grouped[key],axis=0) for key in ids]).T
    assert np.allclose(independent,errors,atol=1e-12,rtol=0)
    saved=np.load(_repo_io(H/'resplit-errors.npz'));assert saved['ids'].tolist()==ids
    assert np.array_equal(saved['mean_squared_errors'],errors)
    pos={str(key):j for j,key in enumerate(d['ids'])};high=np.array([d['noise'][pos[key]]>=12 for key in ids])
    groups={'all':np.ones(len(ids),bool),'low':~high,'high':high}
    result={key:metrics(errors,mask) for key,mask in groups.items()}
    rowerrors=np.array([row[4:] for row in observations]).T
    result['prediction_row_weighting_diagnostic']=metrics(rowerrors,np.ones(rowerrors.shape[1],bool))
    result['prediction_row_weighting_diagnostic']['n_prediction_rows']=result['prediction_row_weighting_diagnostic'].pop('n_unique_ids')
    per_anchor={}
    for k in s.ANCHORS:
        selected=[part for part,row in zip(parts,[(k,j) for k in s.ANCHORS for j in range(5)]) if row[0]==k]
        _,_,rr=p.aggregate(selected);per_anchor[str(k)]=rr
    null_by_group={key:[] for key in groups}
    for permutation in range(s.PERMUTATIONS):
        pieces=[]
        for k,z,y,splits in data:
            rng=np.random.default_rng(2026101611+1000*k+permutation);direction=z['aux'].copy()
            for band in (False,True):
                ix=np.flatnonzero(((z['noise']>=12)==band)&~z['failed']);direction[ix]=direction[rng.permutation(ix)]
            for cal,va in splits:
                pred,_=p.calibrate_predict(z['base'],direction,z['noise'],z['failed'],y,cal,va)
                pieces.append(dict(ids=z['ids'][va].tolist(),base=(z['base'][va]-y[va])**2,aux=(pred-y[va])**2,context=(z['base'][va]-y[va])**2))
        ni,ne,nr=p.aggregate(pieces);assert ni==ids
        assert abs(nr['aux_gain']-source['summary']['alignment_null']['gains'][permutation])<1e-12
        for key,mask in groups.items():null_by_group[key].append(metrics(ne,mask)['aux_gain'])
    for key in groups:
        v=np.array(null_by_group[key]);result[key]['alignment_null']=dict(mean=float(v.mean()),q95=float(np.quantile(v,.95)),
            fraction_at_least_observed=float((1+np.sum(v>=result[key]['aux_gain']))/(len(v)+1)))
        ix=np.flatnonzero(groups[key]);draw=np.random.default_rng(2026101610).choice(ix,(10000,len(ix)),replace=True)
        gain=np.sqrt(errors[0,draw].mean(1))-np.sqrt(errors[1,draw].mean(1))
        result[key]['conditional_ID_bootstrap_gain_ci95']=np.quantile(gain,[.025,.975]).tolist()
    s.save(_repo_io('resplit-subgroups.json'),dict(summary=result,per_anchor=per_anchor,independent_aggregation_verified=True,
        caveat='High-noise subset matches080 support, but remains a diagnostic on historically explored data; no threshold or weights were refitted after inspecting these subgroup outcomes.'))
    np.savez_compressed(_repo_io(H/'resplit-observations.npz'),ids=np.array([x[0] for x in observations]),
                        anchor=np.array([x[1] for x in observations]),split=np.array([x[2] for x in observations]),
                        noise=np.array([x[3] for x in observations]),squared_errors=rowerrors)
    s.check_protected();print(json.dumps(dict(summary=result,per_anchor=per_anchor),indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
