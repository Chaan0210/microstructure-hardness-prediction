"""Frozen-model calibration role stress test; all coefficient fitting stays internal."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import json
import os
import signal
import numpy as np
import torch
from sklearn.model_selection import ShuffleSplit
from threadpoolctl import threadpool_limits
import study as s
import noise_calibration as nc
from evaluate import calibrate

PREVIOUS=_repo_path(s.ROOT, 'experiments/080-followup-20261016')
ANCHORS=(0,1,4)
SPLIT_SEED=2026101608


def extend():
    s.protect();s.sys.addaudithook(s.no_test)
    os.environ['MPLCONFIGDIR']=str(s.HERE/'mpl-cache')
    import full_branch as b
    def timed_out(*unused):raise TimeoutError('Locked15min teacher extension limit')
    signal.signal(signal.SIGALRM,timed_out);signal.alarm(900)
    d,y=b.load();graph=json.loads((s.R4/'label-graph.json').read_text())[4]
    A,B,V=[np.array(graph[k],int) for k in ('A','B','V')]
    b.HERE=s.HERE/'calibration-extension';b.HERE.mkdir(exist_ok=True)
    out=b.fit(d,A,np.r_[V,B[:8]],y[A].copy(),4,'A-query-extension')
    prior=np.load(_repo_io(s.R4/'teachers/fold4-outer0/predictions.npz'))
    assert np.array_equal(prior['query'],B) and np.array_equal(prior['train'],A)
    errors={k:float(np.max(np.abs(out[k][-8:]-prior[k][:8]))) for k in ('base','tab','ridge','logridge','gmm','set')}
    assert max(errors.values())<1e-4,errors
    signal.alarm(0)
    old_g=_repo_path(s.ROOT, 'experiments/residual-cnn-219/main-20261007/fold-4/predictions.npz')
    az=np.load(_repo_io(s.R4/'anchors/fold4.npz'));vz=np.load(_repo_io(old_g))
    assert np.array_equal(d['dev'][vz['V']],V)
    assert np.array_equal(az['A'],A) and np.array_equal(az['B'],B)
    T=np.r_[prior['base'],out['base'][:len(V)]]
    gap=np.r_[prior['tab']-prior['ridge'],(out['tab']-out['ridge'])[:len(V)]]
    G=np.r_[az['prediction_B'],vz['R0']]
    query=np.r_[B,V];base=.16*G+.84*T;failed=d['failed'][query];base[failed]=G[failed]
    keys=list(d['keys']);X=d['X'][query]
    raw=np.c_[T,gap,d['noise'][query],X[:,keys.index('u_ph')],
              np.log1p(np.maximum(np.nan_to_num(X[:,keys.index('n_obj')],nan=0),0)),X[:,keys.index('coverage')]]
    images=s.e.c.load_data()['images'][query]
    preds={};replays={};used=[old_g,s.R4/'anchors/fold4.npz',s.R4/'teachers/fold4-outer0/predictions.npz']
    for family,arm in [('aux','aux'),('context','context_only')]:
        pp=[]
        for j in (0,1):
            path=s.R4/f'{family}-seed{j}/fold4-selector/model.pt';used.append(path)
            ck=torch.load(_repo_io(path),map_location='cpu',weights_only=True);meta=ck['normalization']
            assert meta['train_ids']==d['ids'][A].tolist()
            model=s.e.c.ComplementCNN();model.load_state_dict(ck['state_dict'])
            im=((images-meta['image_mean'])/meta['image_std']).astype(np.float32)
            rr=np.where(np.isfinite(raw),raw,np.array(meta['context_median']))
            ctx=((rr-np.array(meta['context_mean']))/np.array(meta['context_std'])).astype(np.float32)
            pp.append(s.e.c.predict(model,im,ctx,np.arange(len(query)),arm)*meta['target_std'])
        preds[family]=np.mean(pp,axis=0)
        ref=np.load(_repo_io(s.R4/f'{family}-mean2/fold4-selection.npz'))
        assert np.array_equal(ref['B'],B)
        replays[family]=float(np.max(np.abs(preds[family][:len(B)]-ref['residual'])))
        assert replays[family]<1e-4
    np.savez_compressed(_repo_io(s.HERE/'anchor4-heldout187.npz'),query=query,ids=d['ids'][query],A=A,B=B,V=V,
                        base=base,T=T,G=G,noise=d['noise'][query],failed=failed,**preds)
    s.save(_repo_io('calibration-extension-verification.json'),dict(teacher_overlap_max_errors=errors,
        cnn_B_overlap_max_errors=replays,query_ids=d['ids'][query].tolist(),train_ids=d['ids'][A].tolist(),
        disjoint_training_and_query=True,source_sha256={str(p.relative_to(s.ROOT)):s.r.digest(p) for p in used}))
    s.protect();print('ANCHOR4 EXTENSION COMPLETE',errors,replays,flush=True)


def predict(z,y,cal,va,rule,family='aux'):
    eligible=cal[~z['failed'][cal]];direction=z[family]
    if rule=='original':
        w,meta=nc.select(direction[eligible],(y-z['base'])[eligible],z['noise'][eligible]>=12)
        coefficient=float(w[1])
    else:
        high=eligible[z['noise'][eligible]>=12]
        coefficient,raw=calibrate(direction[high],(y-z['base'])[high]);meta=dict(unconstrained=raw)
    high=(z['noise'][va]>=12)&~z['failed'][va]
    pred=z['base'][va]+np.where(high,coefficient*direction[va],0.)
    return pred,dict(coefficient=coefficient,**meta)


def aggregate(parts):
    ids=sorted({i for row in parts for i in row['ids']});pos={x:j for j,x in enumerate(ids)}
    names=('base','original','high_ols','context_original','context_high_ols')
    sums=np.zeros((len(names),len(ids)));counts=np.zeros(len(ids),int)
    for row in parts:
        ix=np.array([pos[i] for i in row['ids']]);counts[ix]+=1
        for j,n in enumerate(names):sums[j,ix]+=row[n]
    return np.array(ids),names,sums/counts,counts


def summary(parts):
    ids,names,errors,counts=aggregate(parts)
    draw=np.random.default_rng(2026101621).integers(0,len(ids),(10000,len(ids)))
    scores={n:float(np.sqrt(errors[j].mean())) for j,n in enumerate(names)}
    increment=np.sqrt(errors[1,draw].mean(1))-np.sqrt(errors[2,draw].mean(1))
    gains=errors[1]-errors[2];order=np.argsort(gains)[::-1];remain=order[10:]
    out=dict(rmse=scores,n_ids=len(ids),prediction_rows=int(counts.sum()),
             high_ols_increment=scores['original']-scores['high_ols'],
             increment_ci95=np.quantile(increment,[.025,.975]).tolist(),
             increment_without_largest10_beneficiaries=float(np.sqrt(errors[1,remain].mean())-np.sqrt(errors[2,remain].mean())),
             largest10_net_gain_share=float(gains[order[:10]].sum()/gains.sum()) if gains.sum()!=0 else None)
    return out,(ids,names,errors,counts)


def evaluate():
    s.protect();d=dict(np.load(_repo_io(s.R4.parent/'inputs.npz')))
    parts=[];records=[];by_anchor={};used=[]
    for k in ANCHORS:
        path=(s.HERE if k==4 else PREVIOUS)/f'anchor{k}-heldout187.npz';used.append(path)
        z=dict(np.load(_repo_io(path)));y=d['y'][z['query']]
        assert set(z['A']).isdisjoint(z['query']) and len(z['query'])==187
        assert np.array_equal(z['ids'],d['ids'][z['query']])
        anchor_parts=[]
        for j,(cal,va) in enumerate(ShuffleSplit(n_splits=5,train_size=107,test_size=80,random_state=SPLIT_SEED).split(y)):
            assert set(cal).isdisjoint(va)
            row=dict(ids=z['ids'][va].tolist(),base=(z['base'][va]-y[va])**2,anchor=k,split=j)
            selections={}
            for family in ('aux','context'):
                for rule in ('original','high_ols'):
                    name=rule if family=='aux' else 'context_'+rule
                    p,meta=predict(z,y,cal,va,rule,family)
                    altered=y.copy();altered[va]+=10000
                    pp,mm=predict(z,altered,cal,va,rule,family)
                    assert np.array_equal(p,pp) and meta==mm
                    row[name]=(p-y[va])**2;selections[name]=meta
            records.append(dict(anchor=k,split=j,calibration_ids=z['ids'][cal].tolist(),evaluation_ids=row['ids'],
                                selection=selections,outer_label_poison_independent=True,
                                new_minus_original_mse=float(np.mean(row['high_ols']-row['original']))))
            parts.append(row);anchor_parts.append(row)
        by_anchor[str(k)]=summary(anchor_parts)[0]
    result,arrays=summary(parts);ids,names,errors,counts=arrays
    passed=all(v['high_ols_increment']>0 for v in by_anchor.values()) and result['increment_ci95'][0]>0 and result['increment_without_largest10_beneficiaries']>0
    s.save(_repo_io('calibration-stress.json'),dict(summary=result,by_anchor=by_anchor,selections=records,
        stronger_production_weight_passes_locked_rule=passed,
        source_sha256={str(p.relative_to(s.ROOT)):s.r.digest(p) for p in used},
        limitation='Conditional reused-image stress test. Anchor4 selected after its large outer gain; not independent confirmation. No Public-derived weight.'))
    np.savez_compressed(_repo_io(s.HERE/'calibration-stress-errors.npz'),ids=ids,names=np.array(names),mean_squared_errors=errors,counts=counts)
    s.protect();print(json.dumps(dict(summary=result,by_anchor=by_anchor,passes=passed),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['extend','evaluate']);a=p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):extend() if a.mode=='extend' else evaluate()
