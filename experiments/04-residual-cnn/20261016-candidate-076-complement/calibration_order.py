"""Compare calibration order and the validation loss's deployed support."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse
import json
import numpy as np
from sklearn.model_selection import ShuffleSplit
from confidence import compare
import probe as p
import noise_calibration as nc
CASES=[(f,o,l) for f in ('legacy','duplicate') for o in ('mean','post') for l in ('original','gated')]
REFERENCE='legacy_mean_original'


def tag(case):return '_'.join(case)


def coefficient(direction,target,band,loss):
    if loss=='original':
        w,meta=nc.select(direction,target,band);return float(w[1]),meta
    if not band.any():return 0.,dict(no_high_rows=True)
    errors=[]
    for strength in nc.STRENGTHS:
        pred=[];truth=[]
        for i in np.flatnonzero(band):
            fit=np.arange(len(target))!=i
            w=nc.fit_weights(direction[fit],target[fit],band[fit],strength)
            pred.append(w[1]*direction[i]);truth.append(target[i])
        errors.append(float(np.mean((np.array(pred)-truth)**2)))
    chosen=nc.STRENGTHS[int(np.argmin(errors))]
    w=nc.fit_weights(direction,target,band,chosen)
    return float(w[1]),dict(strength=chosen,high_only_loo_mse=errors,low_correction_fixed_zero=True)


def correction(rb,rv,target,band,order,loss):
    if order=='mean':
        w,meta=coefficient(rb.mean(0),target,band,loss)
        return w*rv.mean(0),dict(weights=[w],selection=[meta])
    parts=[];weights=[];choices=[]
    for j in range(2):
        w,meta=coefficient(rb[j],target,band,loss);parts.append(w*rv[j]);weights.append(w);choices.append(meta)
    return np.mean(parts,axis=0),dict(weights=weights,selection=choices)


def bands(predictions,base,y):
    rng=np.random.default_rng(2026101637);names=list(predictions);draws=[]
    for _ in range(100):
        ix=rng.integers(0,len(y),(100,len(y)));a=np.sqrt(((base-y)**2)[ix].mean(1))
        draws.append(np.column_stack([a-np.sqrt(((predictions[n]-y)**2)[ix].mean(1)) for n in names]))
    draws=np.concatenate(draws);g=np.array([p.r.rmse(base,y)-p.r.rmse(predictions[n],y) for n in names]);sd=np.maximum(draws.std(0,ddof=1),1e-12)
    critical=float(np.quantile(np.max(np.abs((draws-g)/sd),axis=1),.95))
    return {n:[float(v-critical*s),float(v+critical*s)] for n,v,s in zip(names,g,sd)}


def outer():
    p.protect();d=dict(np.load(_repo_io(p.R4.parent/'inputs.npz')));z=dict(np.load(_repo_io(p.HERE/'baseline076.npz')));fold=z['fold'];y=d['y'][d['dev']]
    out={tag(c):z['p76'].copy() for c in CASES};records=[]
    for k in range(5):
        inner=np.load(_repo_io(p.R4/f'aux-mean2/fold{k}-selection.npz'));B=inner['B'];good=inner['eligible'];ix=np.flatnonzero(fold==k)
        pred={}
        for family in ('legacy','duplicate'):
            pred[family]={}
            for stage in ('selector','final'):
                pp=[]
                for seed in (0,1):
                    path=p.R4/f'aux-seed{seed}/fold{k}-{stage}/predictions.npz' if family=='legacy' else p.HERE/f'stream/duplicate-seed{seed}/fold{k}-{stage}/predictions.npz'
                    a=np.load(_repo_io(path));assert np.array_equal(a['validation_index'],B if stage=='selector' else z['dev'][ix]);pp.append(a['validation_prediction'])
                pred[family][stage]=np.array(pp)
        target=(d['y'][B]-inner['base073'])[good];band=d['noise'][B][good]>=12;active=(z['noise'][ix]>=12)&~z['failed'][ix]
        row=dict(fold=k,selection={})
        for case in CASES:
            family,order,loss=case;rb=pred[family]['selector'][:,good];rv=pred[family]['final']
            rr,meta=correction(rb,rv,target,band,order,loss)
            poisoned=d['y'].copy();poisoned[z['dev'][ix]]+=10000
            check,mm=correction(rb,rv,(poisoned[B]-inner['base073'])[good],band,order,loss)
            assert np.array_equal(rr,check) and meta==mm
            out[tag(case)][ix[active]]+=rr[active];row['selection'][tag(case)]=meta
        records.append(row)
    assert np.max(np.abs(out[REFERENCE]-np.load(_repo_io(p.HERE/'probe-oof.npz'))['old4']))<1e-10
    report={name:dict(rmse=p.r.rmse(pr,y),folds=[p.r.rmse(pr[fold==k],y[fold==k]) for k in range(5)],
                     **compare(out[REFERENCE],pr,y,fold)) for name,pr in out.items()}
    simultaneous=bands({n:v for n,v in out.items() if n!=REFERENCE},out[REFERENCE],y)
    p.save(_repo_io('calibration-order-outer.json'),dict(results=report,selections=records,simultaneous_bands=simultaneous,outer_labels_used_for_fitting=False))
    np.savez_compressed(_repo_io(p.HERE/'calibration-order-oof.npz'),ids=z['ids'],fold=fold,y=y,**out)
    print(json.dumps({n:dict(rmse=v['rmse'],gain=v['gain'],ci95=v['ci95']) for n,v in report.items()},indent=2),flush=True)
    p.protect()


def frozen_predictions():
    import torch
    from PIL import Image
    import stream_cnn as s
    torch.set_num_threads(1);p.protect()
    src=np.load(_repo_io(p.R4.parent/'inputs.npz'));X=src['X'];keys=list(src['keys']);ids=src['ids'];records=[]
    for k in (0,1,4):
        previous=_repo_path(p.ROOT, f'experiments/{"078-080-update-20261016" if k==4 else "080-followup-20261016"}')
        z=dict(np.load(_repo_io(previous/f'anchor{k}-heldout187.npz')));B,V,A=z['B'],z['V'],z['A'];query=z['query']
        assert np.array_equal(query,np.r_[B,V]) and set(A).isdisjoint(query)
        inner=np.load(_repo_io(p.R4/f'teachers/fold{k}-outer0/predictions.npz'))
        extension=previous/('calibration-extension' if k==4 else 'extensions')/f'teachers/fold{k}-A-query-extension/predictions.npz'
        ext=np.load(_repo_io(extension));assert np.array_equal(ext['query'][:len(V)],V)
        assert np.array_equal(inner['query'],B) and np.array_equal(inner['train'],A) and np.array_equal(ext['train'],A)
        gap=np.r_[inner['tab']-inner['ridge'],(ext['tab']-ext['ridge'])[:len(V)]]
        xx=X[query];raw=np.c_[z['T'],gap,z['noise'],xx[:,keys.index('u_ph')],
            np.log1p(np.maximum(np.nan_to_num(xx[:,keys.index('n_obj')],nan=0),0)),xx[:,keys.index('coverage')]]
        images=np.stack([np.asarray(Image.open(_repo_path(p.ROOT, f'data/train/{i}.png')),np.float32)/255. for i in ids[query]])[:,None]
        predictions={};checks={}
        for family in ('legacy','duplicate'):
            ps=[];checks[family]=[]
            for seed in (0,1):
                path=p.R4/f'aux-seed{seed}/fold{k}-selector/model.pt' if family=='legacy' else p.HERE/f'stream/duplicate-seed{seed}/fold{k}-selector/model.pt'
                ck=torch.load(_repo_io(path),map_location='cpu',weights_only=True);meta=ck['normalization']
                assert set(meta['train_ids'])==set(ids[A])
                model=s.Original() if family=='legacy' else s.TwoStream('duplicate');model.load_state_dict(ck['state_dict'])
                im=images.copy();im-=meta['image_mean'];im/=meta['image_std']
                rr=np.where(np.isfinite(raw),raw,np.array(meta['context_median']))
                ctx=((rr-np.array(meta['context_mean']))/np.array(meta['context_std'])).astype(np.float32)
                pred=s.e.c.predict(model,im,ctx,np.arange(len(query)),'aux')*meta['target_std'];ps.append(pred)
                ref=np.load(_repo_io(path.with_name('predictions.npz')));assert np.array_equal(ref['validation_index'],B)
                error=float(np.max(np.abs(pred[:len(B)]-ref['validation_prediction'])));assert error<1e-4,error
                checks[family].append(dict(seed=seed,B_replay_max_error=error,model_sha256=p.r.digest(path)))
            predictions[family]=np.array(ps)
        if 'aux' in z:assert np.max(np.abs(predictions['legacy'].mean(0)-z['aux']))<1e-4
        out=p.HERE/f'calibration-roles/anchor{k}.npz';out.parent.mkdir(exist_ok=True)
        np.savez_compressed(_repo_io(out),query=query,ids=z['ids'],base=z['base'],noise=z['noise'],failed=z['failed'],A=A,**predictions)
        records.append(dict(anchor=k,checks=checks,base_source_sha256=p.r.digest(previous/f'anchor{k}-heldout187.npz'),
                            extension_sha256=p.r.digest(extension),all_query_rows_excluded_from_model_training=True))
        print('FROZEN ROLE PREDICTIONS',k,flush=True)
    p.save(_repo_io('calibration-role-prediction-verification.json'),dict(all_passed=True,anchors=records,teacher_baseline='strict073 A-only'))
    p.protect()


def summarize_roles(parts):
    names=[tag(c) for c in CASES];ids=sorted({q for row in parts for q in row['ids']});pos={q:j for j,q in enumerate(ids)}
    sums=np.zeros((len(names),len(ids)));counts=np.zeros(len(ids),int)
    for row in parts:
        ix=np.array([pos[q] for q in row['ids']]);counts[ix]+=1;sums[:,ix]+=row['squared_errors']
    errors=sums/counts;ref=names.index(REFERENCE);rng=np.random.default_rng(2026101641);draws=[]
    for _ in range(100):
        ix=rng.integers(0,len(ids),(100,len(ids)))
        scores=np.sqrt(errors[:,ix].mean(2));draws.append((scores[ref]-scores).T)
    draws=np.concatenate(draws);scores=np.sqrt(errors.mean(1));gains=scores[ref]-scores
    sd=np.maximum(draws.std(0,ddof=1),1e-12);critical=float(np.quantile(np.max(np.abs((draws-gains)/sd),axis=1),.95))
    results={}
    for j,name in enumerate(names):
        order=np.argsort(errors[ref]-errors[j])[::-1];keep=order[10:]
        results[name]=dict(rmse=float(scores[j]),gain=float(gains[j]),ci95=np.quantile(draws[:,j],[.025,.975]).tolist(),
            simultaneous_band=[float(gains[j]-critical*sd[j]),float(gains[j]+critical*sd[j])],
            gain_without_largest10=float(np.sqrt(errors[ref,keep].mean())-np.sqrt(errors[j,keep].mean())))
    return dict(n_unique_ids=len(ids),prediction_rows=int(counts.sum()),results=results),(ids,names,errors,counts)


def stress():
    p.protect();d=dict(np.load(_repo_io(p.R4.parent/'inputs.npz')));parts=[];records=[];by_anchor={}
    for k in (0,1,4):
        z=dict(np.load(_repo_io(p.HERE/f'calibration-roles/anchor{k}.npz')));y=d['y'][z['query']];local=[]
        assert set(z['A']).isdisjoint(z['query'])
        for j,(cal,va) in enumerate(ShuffleSplit(n_splits=5,train_size=107,test_size=80,random_state=2026101608).split(y)):
            good=cal[~z['failed'][cal]];band=z['noise'][good]>=12;active=(z['noise'][va]>=12)&~z['failed'][va]
            target=(y-z['base'])[good];errors=[];selection={}
            for case in CASES:
                family,order,loss=case;rb=z[family][:,good];rv=z[family][:,va]
                rr,meta=correction(rb,rv,target,band,order,loss)
                changed=y.copy();changed[va]+=10000
                check,mm=correction(rb,rv,(changed-z['base'])[good],band,order,loss)
                assert np.array_equal(rr,check) and meta==mm
                pred=z['base'][va]+np.where(active,rr,0.);errors.append((pred-y[va])**2);selection[tag(case)]=meta
            row=dict(ids=z['ids'][va].tolist(),squared_errors=np.array(errors));parts.append(row);local.append(row)
            records.append(dict(anchor=k,split=j,calibration_ids=z['ids'][cal].tolist(),evaluation_ids=z['ids'][va].tolist(),selection=selection,outer_poison_exact=True))
        by_anchor[str(k)]=summarize_roles(local)[0]
        print('CALIBRATION ROLE TEST',k,flush=True)
    report,arrays=summarize_roles(parts);ids,names,errors,counts=arrays
    previous=json.loads((p.PREV/'calibration-stress.json').read_text())['summary']['rmse']['original']
    assert abs(report['results'][REFERENCE]['rmse']-previous)<1e-5
    outer_report=json.loads((p.HERE/'calibration-order-outer.json').read_text());adopt=[]
    for name,rr in report['results'].items():
        if name==REFERENCE:continue
        oo=outer_report['results'][name]
        outer_pass=oo['gain']>=.01 and sum(x>1e-10 for x in oo['fold_gains'])>=3 and oo['ci95'][0]>0 and outer_report['simultaneous_bands'][name][0]>0
        stress_pass=rr['ci95'][0]>0 and rr['simultaneous_band'][0]>0 and rr['gain_without_largest10']>0 and all(a['results'][name]['gain']>0 for a in by_anchor.values())
        if outer_pass and stress_pass:adopt.append(name)
    p.save(_repo_io('calibration-order-stress.json'),dict(summary=report,by_anchor=by_anchor,selections=records,adopt=adopt,
        reference_reproduced=True,teacher_baseline='frozen A-only strict073',
        limitation='Previously explored images and three selected anchors; calibration stability test, not fresh full076 generalization.'))
    np.savez_compressed(_repo_io(p.HERE/'calibration-order-stress-errors.npz'),ids=np.array(ids),names=np.array(names),mean_squared_errors=errors,counts=counts)
    print(json.dumps(dict(summary=report,anchor_gains={k:{n:v['gain'] for n,v in a['results'].items()} for k,a in by_anchor.items()},adopt=adopt),indent=2),flush=True)
    p.protect()


if __name__=='__main__':
    import torch
    from threadpoolctl import threadpool_limits
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['outer','predictions','stress','all']);args=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.mode=='all':outer();frozen_predictions();stress()
        else:{'outer':outer,'predictions':frozen_predictions,'stress':stress}[args.mode]()
