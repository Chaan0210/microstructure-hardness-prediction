"""Read-only checkpoint replay and conditional resplit diagnostics for080."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
sys.dont_write_bytecode=True
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import ShuffleSplit
from threadpoolctl import threadpool_limits
import study as s
import cnn_experiment as e
import noise_calibration as nc
H=s.HERE;R4=s.R4;ROOT=s.ROOT


def teacher_oof(d,k,kind):
    g=json.loads((R4/'label-graph.json').read_text())[k]
    values=np.full(500,np.nan);gap=np.full(500,np.nan)
    for row in g['nodes']:
        if row['name'].startswith(kind):
            z=np.load(_repo_io(R4/f"teachers/fold{k}-{row['name']}/predictions.npz"))
            values[z['query']]=z['base'];gap[z['query']]=z['tab']-z['ridge']
    return values,gap


def profile():
    d=s.load_inputs();y=d['y'];third=np.full(500,np.nan);thirdgap=np.full(500,np.nan)
    for k in range(3):
        z=np.load(_repo_io(R4/f'final500/teachers/fold{k}-full500/predictions.npz'))
        third[z['query']]=z['base'];thirdgap[z['query']]=z['tab']-z['ridge']
    graphs=json.loads((R4/'label-graph.json').read_text());rows=[]
    for k in s.ANCHORS:
        A=np.array(graphs[k]['A']);small,smallgap=teacher_oof(d,k,'deep');middle,midgap=teacher_oof(d,k,'outer')
        comparisons={}
        for band in (False,True):
            ix=A[(d['noise'][A]>=12)==band]
            cases={}
            for name,p in [('teacher142',small),('teacher213',middle),('teacher333',third)]:
                z=y[ix]-p[ix];cases[name]=dict(n=len(ix),mean=float(z.mean()),sd=float(z.std()),rms=float(np.sqrt(np.mean(z*z))))
            cases['paired_prediction_shift_rms']={name:float(np.sqrt(np.mean((p[ix]-small[ix])**2))) for name,p in [('213_vs142',middle),('333_vs142',third)]}
            comparisons['high' if band else 'low']=cases
        norms={}
        for stage,path in [('A213',R4/f'aux-seed0/fold{k}-selector/model.pt'),('T320',R4/f'aux-seed0/fold{k}-final/model.pt'),('full500',R4/'final500/models/aux-seed0/model.pt')]:
            c=torch.load(_repo_io(path),map_location='cpu',weights_only=True);n=c['normalization']
            norms[stage]={key:n[key] for key in ('image_mean','image_std','target_mean','target_std','context_mean','context_std')}
            norms[stage].update(n_train=len(n['train_ids']),epochs=c['epochs'],updates=c['epochs']*int(np.ceil(len(n['train_ids'])/32)))
        rows.append(dict(anchor=k,common_query_ids=d['ids'][A].tolist(),target_profiles=comparisons,actual_normalization=norms))
    weights=json.loads((R4/'noise-calibration/summary.json').read_text())['aux']['selection']
    summary=dict(anchors=rows,production_weights=json.loads((R4/'final500/calibration-frozen.json').read_text()),
        fold_weights=[dict(fold=x['fold'],weights=x['addition']['weights'],strength=x['addition']['strength']) for x in weights],
        limitations=['142/213/333 teachers differ in training sample count AND composition; this is actual-procedure drift, not isolated sample-size causality.',
                     'Full500 training predictions are not held-out validation.',
                     'Production mean coefficients have no separate honest OOF score; they must not be applied globally to old OOF and called cross-validation.'])
    s.save(_repo_io('transport-profile.json'),summary);print('TARGET AND NORMALIZATION PROFILES COMPLETE',flush=True)


def raw_context(d,query,T,gap):
    keys=list(d['keys'])
    return np.c_[T,gap,d['noise'][query],d['X'][query,keys.index('u_ph')],
        np.log1p(np.maximum(np.nan_to_num(d['X'][query,keys.index('n_obj')],nan=0),0)),d['X'][query,keys.index('coverage')]]


def predict(folder,k,stage,images,raw):
    pred=[];paths=[]
    for seed in range(2):
        path=R4/f'{folder}-seed{seed}/fold{k}-{stage}/model.pt';paths.append(path)
        checkpoint=torch.load(_repo_io(path),map_location='cpu',weights_only=True);meta=checkpoint['normalization']
        model=e.c.ComplementCNN();model.load_state_dict(checkpoint['state_dict'])
        xx=((images-meta['image_mean'])/meta['image_std']).astype(np.float32)
        rr=np.where(np.isfinite(raw),raw,np.array(meta['context_median']))
        context=((rr-np.array(meta['context_mean']))/np.array(meta['context_std'])).astype(np.float32)
        pred.append(e.c.predict(model,xx,context,np.arange(len(images)),checkpoint['arm'])*meta['target_std'])
    s.protect(paths);return np.mean(pred,axis=0)


def legacy_anchor(k,d):
    # The old predictor is loaded only for inference; no old entry point is run.
    old=_repo_path(ROOT, 'experiments/residual-cnn-219');sys.path.insert(0,str(old))
    from data import load_development
    import joblib
    features,_,contract=load_development();assert np.array_equal(features.ids,d['ids'][d['dev']])
    graph=contract['graphs'][k];A,B,V=[np.array(graph[key]) for key in ('A','B','V')]
    path=old/f'main-20261007/fold-{k}/frozen-anchor.joblib'
    binding=json.loads(path.with_suffix('.json').read_text());assert s.r.digest(path)==binding['anchor_sha256']
    model=joblib.load(_repo_io(path));assert list(model.train)==A.tolist()
    query=np.r_[B,V];combined=model.predict(features,query)
    prediction=np.r_[model.predict(features,B),model.predict(features,V)]
    batch_error=float(np.max(np.abs(prediction-combined)));assert batch_error<1e-4,batch_error
    s.save(_repo_io(f'anchor{k}-legacy-query-check.json'),dict(combined_vs_original_batches_max_error=batch_error,
        original_B_and_V_batching_used=True))
    b=np.load(_repo_io(R4/f'anchors/fold{k}.npz'));oldpred=np.load(_repo_io(path.parent/'predictions.npz'))
    assert np.array_equal(prediction[:len(B)],b['prediction_B'])
    assert np.array_equal(oldpred['V'],V) and np.max(np.abs(prediction[len(B):]-oldpred['R0']))<1e-8
    s.protect([path,path.with_suffix('.json'),path.parent/'predictions.npz',R4/f'anchors/fold{k}.npz'])
    return d['dev'][query],prediction


def replay():
    d=e.c.load_data();y=d['y'];graphs=json.loads((R4/'label-graph.json').read_text())
    strict=np.load(_repo_io(R4.parent/'scope.npz'))['strict_blend'][0]
    ridge=np.load(_repo_io(R4.parent/'consistency-strict.npz'))['production'][0]
    oldweights=json.loads((R4/'noise-calibration/summary.json').read_text())['aux']['selection'];records=[]
    for k in s.ANCHORS:
        g=graphs[k];A,B,V=[np.array(g[key]) for key in ('A','B','V')];O=np.r_[B,V]
        prior=np.load(_repo_io(R4/f'teachers/fold{k}-outer0/predictions.npz'))
        extended=np.load(_repo_io(H/f'extensions/teachers/fold{k}-A-query-extension/predictions.npz'))
        assert np.array_equal(extended['query'][:len(V)],V)
        T=np.r_[prior['base'],extended['base'][:len(V)]]
        gap=np.r_[prior['tab']-prior['ridge'],extended['tab'][:len(V)]-extended['ridge'][:len(V)]]
        query,G=legacy_anchor(k,d);assert np.array_equal(query,O)
        P=.16*G+.84*T;bad=np.nan_to_num(d['X'][O,list(d['keys']).index('n_obj')],nan=0)<20;P[bad]=G[bad]
        raw=raw_context(d,O,T,gap);images=d['images'][O]
        aux=predict('aux',k,'selector',images,raw);ctx=predict('context',k,'selector',images,raw)
        for family,p in [('aux',aux),('context',ctx)]:
            ref=np.load(_repo_io(R4/f'{family}-mean2/fold{k}-selection.npz'));assert np.array_equal(ref['B'],B)
            assert np.max(np.abs(ref['residual']-p[:len(B)]))<1e-4
        np.savez_compressed(_repo_io(H/f'anchor{k}-heldout187.npz'),query=O,ids=d['ids'][O],A=A,B=B,V=V,
                            base=P,T=T,G=G,aux=aux,context=ctx,noise=d['noise'][O],failed=bad)
        ix=np.array([list(d['dev']).index(i) for i in V]);baseT=strict[ix]
        TT=(baseT-.16*d['C73_cnn'][0,ix])/.84
        rawT=raw_context(d,V,TT,d['C73_tab'][0,ix]-ridge[ix])
        crossed=predict('aux',k,'selector',d['images'][V],rawT)
        final=predict('aux',k,'final',d['images'][V],rawT)
        expected=np.load(_repo_io(R4/f'aux-mean2/fold{k}-outer.npz'))['residual'];failed=d['fail'][0,ix]
        assert np.max(np.abs(final[~failed]-expected[~failed]))<1e-4
        weights=np.array(oldweights[k]['addition']['weights']);w=weights[(d['noise'][V]>=12).astype(int)];w[failed]=0.
        nativeA=P[len(B):]+w*aux[len(B):];cross=baseT+w*crossed;nativeT=baseT+w*final
        records.append(dict(anchor=k,n_query=len(V),weights_frozen_from_original_B=weights.tolist(),
            A_base_A_CNN=s.metric(P[len(B):],nativeA,y[V]),
            T_base_A_CNN=s.metric(baseT,cross,y[V]),T_base_T_CNN=s.metric(baseT,nativeT,y[V]),
            same_base_CNN_refit_effect=s.metric(cross,nativeT,y[V]),
            raw_context_change_rms=float(np.sqrt(np.mean((crossed-aux[len(B):])**2))),
            raw_CNN_refit_change_rms=float(np.sqrt(np.mean((final-crossed)**2)))))
        np.savez_compressed(_repo_io(H/f'anchor{k}-refit-contrast.npz'),ids=d['ids'][V],query=V,baseA=P[len(B):],baseT=baseT,
                            A_A=nativeA,T_A=cross,T_T=nativeT,rawA=aux[len(B):],raw_cross=crossed,rawT=final)
        s.save(_repo_io('refit-replay.json'),dict(anchors=records,scope='Two fixed pilot anchors; sameV80 within each contrast, not a new independent data sample.',
            interpretation='T/A toT/T combines teacher targets, CNN training size and preprocessing/optimizer changes. It does not isolate those mechanisms individually.'))
        print('PAIRED REPLAY COMPLETE',k,flush=True)
    s.check_protected()


def calibrate_predict(base,direction,noise,failed,y,cal,valid):
    train=cal[~failed[cal]];band=noise[train]>=12
    weights,meta=nc.select(direction[train],(y-base)[train],band)
    pred=base[valid]+weights[(noise[valid]>=12).astype(int)]*direction[valid]
    pred[failed[valid]]=base[valid][failed[valid]]
    return pred,dict(weights=weights.tolist(),**meta)


def aggregate(rows):
    ids=sorted(set(key for part in rows for key in part['ids']))
    errors={key:[] for key in ('base','aux','context')};lookup={key:j for j,key in enumerate(ids)}
    sums=np.zeros((3,len(ids)));counts=np.zeros(len(ids))
    for part in rows:
        ix=np.array([lookup[key] for key in part['ids']]);counts[ix]+=1
        for j,key in enumerate(('base','aux','context')):sums[j,ix]+=part[key]
    mean=sums/counts
    return ids,mean,dict(n_unique_ids=len(ids),n_prediction_rows=int(counts.sum()),
        base_rmse=float(np.sqrt(mean[0].mean())),aux_rmse=float(np.sqrt(mean[1].mean())),context_rmse=float(np.sqrt(mean[2].mean())),
        aux_gain=float(np.sqrt(mean[0].mean())-np.sqrt(mean[1].mean())),
        image_increment=float(np.sqrt(mean[2].mean())-np.sqrt(mean[1].mean())))


def resplit():
    d=s.load_inputs();records=[];parts=[];datasets=[]
    for k in s.ANCHORS:
        z=dict(np.load(_repo_io(H/f'anchor{k}-heldout187.npz')));y=d['y'][z['query']]
        assert set(z['A']).isdisjoint(z['query']) and len(z['query'])==187
        splits=list(ShuffleSplit(n_splits=5,train_size=107,test_size=80,random_state=s.SPLIT_SEED).split(y))
        datasets.append((k,z,y,splits))
        for j,(cal,va) in enumerate(splits):
            out={};meta={}
            for family in ('aux','context'):
                out[family],meta[family]=calibrate_predict(z['base'],z[family],z['noise'],z['failed'],y,cal,va)
                poison=y.copy();poison[va]+=10000
                pp,mm=calibrate_predict(z['base'],z[family],z['noise'],z['failed'],poison,cal,va)
                assert np.array_equal(pp,out[family]) and mm==meta[family]
            records.append(dict(anchor=k,split=j,calibration_ids=z['ids'][cal].tolist(),evaluation_ids=z['ids'][va].tolist(),
                aux=s.metric(z['base'][va],out['aux'],y[va]),context=s.metric(z['base'][va],out['context'],y[va]),
                image_increment=s.metric(out['context'],out['aux'],y[va]),selection=meta,evaluation_label_poison_exact=True))
            parts.append(dict(ids=z['ids'][va].tolist(),base=(z['base'][va]-y[va])**2,aux=(out['aux']-y[va])**2,context=(out['context']-y[va])**2))
    ids,errors,result=aggregate(parts)
    draw=np.random.default_rng(2026101610).integers(0,len(ids),(10000,len(ids)))
    gain=np.sqrt(errors[0,draw].mean(1))-np.sqrt(errors[1,draw].mean(1))
    increment=np.sqrt(errors[2,draw].mean(1))-np.sqrt(errors[1,draw].mean(1))
    result['conditional_ID_bootstrap_gain_ci95']=np.quantile(gain,[.025,.975]).tolist()
    result['conditional_ID_bootstrap_image_increment_ci95']=np.quantile(increment,[.025,.975]).tolist()
    null=[]
    for permutation in range(s.PERMUTATIONS):
        parts_null=[]
        for k,z,y,splits in datasets:
            rng=np.random.default_rng(2026101611+1000*k+permutation);direction=z['aux'].copy()
            for band in (False,True):
                ix=np.flatnonzero(((z['noise']>=12)==band)&~z['failed']);direction[ix]=direction[rng.permutation(ix)]
            for cal,va in splits:
                pred,_=calibrate_predict(z['base'],direction,z['noise'],z['failed'],y,cal,va)
                parts_null.append(dict(ids=z['ids'][va].tolist(),base=(z['base'][va]-y[va])**2,aux=(pred-y[va])**2,context=(z['base'][va]-y[va])**2))
        _,_,rr=aggregate(parts_null);null.append(rr['aux_gain'])
        if (permutation+1)%8==0:print('ALIGNMENT NULL',permutation+1,'/',s.PERMUTATIONS,flush=True)
    result['alignment_null']=dict(n=s.PERMUTATIONS,gains=null,mean=float(np.mean(null)),q95=float(np.quantile(null,.95)),
        fraction_at_least_observed=float((1+np.sum(np.array(null)>=result['aux_gain']))/(1+len(null))),
        interpretation='Within-noise alignment stress test. Not conditional causal information, not historical selection correction, not independent new-data evidence.')
    result['scope']='FixedA-only models; new107/80 calibration/evaluation roles on their187 excluded images. Images were previously explored elsewhere.'
    s.save(_repo_io('resplit-results.json'),dict(summary=result,splits=records))
    np.savez_compressed(_repo_io(H/'resplit-errors.npz'),ids=np.array(ids),mean_squared_errors=errors)
    s.check_protected();print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('profile','replay','resplit'));a=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):{'profile':profile,'replay':replay,'resplit':resplit}[a.mode]()
