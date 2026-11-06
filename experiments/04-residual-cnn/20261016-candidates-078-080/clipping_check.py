"""Train-fold-only range clipping check for the existing078 H9 partners."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import study as s


def main():
    s.protect();s.sys.addaudithook(s.no_test)
    h=_repo_path(s.ROOT, 'experiments/push-1300');d=dict(np.load(_repo_io(s.R4.parent/'inputs.npz')))
    ids=d['ids'];pos={str(i):j for j,i in enumerate(ids)};dev=d['dev'];y=d['y']
    c=np.load(_repo_io(h/'cumul-train.npz'));cpos={str(i):j for j,i in enumerate(c['ids'])}
    extra=c['X'][[cpos[str(i)] for i in ids]][:,[1,4]]
    g=json.loads((h/'abl/groups.json').read_text());drop=set(g['intensity'].split()+g['het'].split())
    variants=('u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z')
    features={}
    for v in variants:
        z=np.load(_repo_path(s.ROOT, f'experiments/push-800/feat-{v}-anghcor-train.npz'),allow_pickle=True)
        ip={str(q):j for j,q in enumerate(z['ids'])};keep=[j for j,k in enumerate(z['keys']) if str(k) not in drop]
        features[v]=z['X'][[ip[str(i)] for i in ids]].astype(float)[:,keep]
    z=np.load(_repo_io(s.R4.parent/'round2/gmm-features500.npz'));assert np.array_equal(z['ids'],ids)
    gmmx=np.c_[features['u1dmavgH2Z'],z['X']]
    source=_repo_path(s.ROOT, 'experiments/push-900/gmm_partner.py')
    text=source.read_text().split('mode = sys.argv[1]')[0]
    gp={'__file__':str(source),'__name__':'gmm_clipping_reuse'};exec(compile(text,str(source),'exec'),gp)
    paths=['resolution-216/ensemble-OOF.npz','resolution-repartition-217/20261031/ensemble-OOF.npz',
           'resolution-repartition-217/20261101/ensemble-OOF.npz']
    out={key:np.full((3,400),np.nan) for key in ('ridge_raw','ridge_clip','logridge_raw','logridge_clip','gmm_raw','gmm_clip')}
    fold_records=[];devpos={int(q):j for j,q in enumerate(dev)}
    for partition,path in enumerate(paths):
        z=np.load(_repo_path(s.ROOT, 'experiments', path));indices=np.array([pos[str(i)] for i in z['ids']])
        for k in range(5):
            tr=indices[z['fold']!=k];va=indices[z['fold']==k];vi=np.array([devpos[int(i)] for i in va])
            ex=np.clip(extra[va],extra[tr].min(0),extra[tr].max(0))
            pred={}
            for name,X in [('ridge',features['u1dmavgH2Z']),('gmm',gmmx)]+[('log_'+v,features[v]) for v in variants]:
                xt=np.c_[X[tr],extra[tr]];xq=np.r_[np.c_[X[va],extra[va]],np.c_[X[va],ex]]
                if name=='gmm':p,alpha=gp['fit_predict'](xt,y[tr],xq,19001)
                else:
                    islog=name.startswith('log_')
                    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                                        Ridge(alpha=2.) if islog else RidgeCV(alphas=[.1,1,10,100,1000]))
                    p=model.fit(xt,np.log(y[tr]) if islog else y[tr]).predict(xq)
                    if islog:p=np.exp(p)
                pred[name]=p
            pred['logridge']=np.exp(np.mean([np.log(pred['log_'+v]) for v in variants],axis=0))
            for name in ('ridge','logridge','gmm'):
                out[name+'_raw'][partition,vi]=pred[name][:len(va)]
                out[name+'_clip'][partition,vi]=pred[name][len(va):]
            fold_records.append(dict(partition=partition,fold=k,n_clipped=int(np.any(ex!=extra[va],axis=1).sum()),
                                     n_clipped_low=int((np.any(ex!=extra[va],axis=1)&(d['noise'][va]<12)).sum()),
                                     training_ids=ids[tr].tolist(),validation_ids=ids[va].tolist()))
    reference=np.load(_repo_io(h/'h9-partners.npz'));errors={name:float(np.max(np.abs(out[name+'_raw']-reference['phys_'+name]))) for name in ('ridge','logridge','gmm')}
    assert max(errors.values())<1e-5,errors
    old=np.load(_repo_io(h/'dev-combo.npy'));ydev=y[dev];gate=(d['noise'][dev]<12)&~d['fail']
    preds={}
    for mode in ('raw','clip'):
        delta=sum(w*(out[n+'_'+mode]-reference['none_'+n]) for n,w in [('ridge',.128),('logridge',.128),('gmm',.15)])
        preds[mode]=old+np.where(gate,delta,0.)
    records={name:dict(pooled_rmse=float(np.sqrt(np.mean((p-ydev)**2))),
                       partition_rmse=np.sqrt(np.mean((p-ydev)**2,axis=1)).tolist()) for name,p in [('076',old),*preds.items()]}
    s.save(_repo_io('clipping-check.json'),dict(scores=records,raw_reproduction_errors=errors,folds=fold_records,
                                     clipping_gain=float(records['raw']['pooled_rmse']-records['clip']['pooled_rmse']),
                                     test_inputs_used=False,interpretation='Diagnostic of a single preprocessing mismatch; no Public causal identification.'))
    np.savez_compressed(_repo_io(s.HERE/'clipping-check.npz'),ids=ids[dev],fold=d['fold'],y=ydev,**out,**{'blend_'+k:v for k,v in preds.items()})
    s.protect();print(json.dumps(dict(scores=records,raw_reproduction_errors=errors),indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
