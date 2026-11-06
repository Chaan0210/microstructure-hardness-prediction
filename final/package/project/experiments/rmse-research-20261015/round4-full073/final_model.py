"""Fit the selected full500 CNNs, then separately infer experimental CSVs."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'2')
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
import cnn_experiment as e
from final_teachers import forbid_test_inputs
HERE=e.HERE;FINAL=HERE/'final500';r=e.r
SEEDS=(2026101501,2026201501)


def protected():
    manifest=json.loads((HERE.parent/'provenance.json').read_text())
    saved={k:v for k,v in manifest.items() if '/submission-candidate-' in k}
    for name,digest in saved.items():assert r.digest(r.ROOT/name)==digest,name
    return saved


def load_fitter():
    source=HERE/'cnn_fitter.py';text=source.read_text()
    needle='float(np.max(np.abs(rp-vp)))';assert text.count(needle)==1
    text=text.replace(needle,'float(np.max(np.abs(rp-vp), initial=0.))')
    text='# Full500 variant: only empty-query reload diagnostics change.\n'+text
    path=HERE/'full500_fitter.py'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    spec=importlib.util.spec_from_file_location('full500_cnn_fitter',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    r.save_json(FINAL/'fitter-derivation.json',dict(source_sha256=r.digest(source),derived_sha256=r.digest(path),
                                                 learning_operations_unchanged=True,only_empty_query_diagnostics_changed=True))
    return module


def teacher_inputs(d):
    plan=json.loads((FINAL/'teacher-plan.json').read_text());base=np.full(500,np.nan);gap=np.full(500,np.nan)
    count=np.zeros(500,int);sources=[]
    assert plan['training_ids']==d['ids'].tolist()
    for row in plan['folds']:
        path=FINAL/f"teachers/fold{row['fold']}-full500/predictions.npz";z=np.load(path)
        tr,va=z['train'],z['query']
        assert np.array_equal(tr,row['train']) and np.array_equal(va,row['query'])
        assert set(tr).isdisjoint(va) and set(tr)|set(va)==set(range(500))
        base[va]=z['base'];gap[va]=z['tab']-z['ridge'];count[va]+=1
        sources.append(dict(train_ids=d['ids'][tr].tolist(),validation_ids=d['ids'][va].tolist(),
                            file=str(path.relative_to(HERE)),sha256=r.digest(path)))
    assert np.all(count==1) and np.isfinite(base).all() and np.isfinite(gap).all()
    keys=list(d['keys']);context=np.c_[base,gap,d['noise'],d['X'][:,keys.index('u_ph')],
        np.log1p(np.maximum(np.nan_to_num(d['X'][:,keys.index('n_obj')],nan=0),0)),d['X'][:,keys.index('coverage')]]
    return base,context,sources


def verify_teachers(d,labels):
    records=[];plan=json.loads((FINAL/'teacher-plan.json').read_text())
    variants=('u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z')
    for row in plan['folds']:
        folder=FINAL/f"teachers/fold{row['fold']}-full500";z=np.load(folder/'predictions.npz')
        tr=z['train'];query=z['query'];contract=json.loads((folder/'contract.json').read_text())
        assert contract['target_sha256']==hashlib.sha256(labels[tr].tobytes()).hexdigest()
        assert contract['train_ids']==d['ids'][tr].tolist() and contract['query_ids']==d['ids'][query].tolist()
        for band in (0,1):
            mask=(d['noise'][query]>=12)==bool(band);qq=query[mask];pp=[]
            for slot in ('038','039','044'):
                for variant in variants:
                    for seed in range(3):
                        a=np.load(folder/f'tab-{band}-{slot}-{variant}-seed{seed}.npz')
                        assert np.array_equal(a['train'],tr) and np.array_equal(a['query'],qq);pp.append(a['prediction'])
            assert np.array_equal(np.mean(pp,axis=0),z['tab'][mask])
        sets=[]
        for seed in (731503,731509,731527):
            a=np.load(folder/f'grain-set-{seed}.npz');assert np.array_equal(a['train'],tr) and np.array_equal(a['query'],query)
            sets.append(a['prediction']);stopping=json.loads((folder/f'grain-set-{seed}.json').read_text())['stopping']
            aa,bb=set(stopping['inner_train_ids']),set(stopping['inner_validation_ids'])
            assert not aa&bb and aa|bb==set(d['ids'][tr])
        assert np.array_equal(np.mean(sets,axis=0),z['set'])
        detail=json.loads((folder/'details.json').read_text())
        for item in detail['linear']:assert set(item['augmentation_source_ids'])<=set(d['ids'][tr])
        rebuilt=(.384*z['tab']+.128*z['ridge']+.128*z['logridge']+.05*z['set']+.15*z['gmm'])/.84
        failed=np.nan_to_num(d['X'][query,list(d['keys']).index('n_obj')],nan=0)<20
        rebuilt[failed]=labels[tr].mean();assert np.array_equal(rebuilt,z['base'])
        records.append(dict(fold=row['fold'],n_train=len(tr),n_query=len(query),all72_tab_fits_present=True,
                            all3_grain_set_fits_present=True,strict_source_scope=True,component_sum_exact=True))
    r.save_json(FINAL/'teacher-verification.json',dict(all_passed=True,folds=records))


def fit():
    assert json.loads((FINAL/'teachers-complete.json').read_text())['complete']
    sys.addaudithook(forbid_test_inputs)
    original=protected();d=e.c.load_data();labels=d.pop('y');train=np.arange(500)
    assert len(set(d['ids']))==500 and all(str(i).startswith('TRAIN_') for i in d['ids'])
    verify_teachers(d,labels)
    base,raw,sources=teacher_inputs(d)
    data=e.cnn_data(d,train,np.array([],int),labels.copy(),base,sources,-1,'full500_refit',raw)
    data['meta'].update(test_inputs_read=False,teacher_split_seed=2026101607,validation_role='none; full training fit')
    weights_source=HERE/'noise-calibration/summary.json'
    choices=json.loads(weights_source.read_text())['aux']['selection']
    weights={arm:np.mean([row[arm]['weights'] for row in choices],axis=0).tolist() for arm in ('addition','replacement')}
    frozen_path=FINAL/'calibration-frozen.json'
    if not frozen_path.exists():
        r.save_json(frozen_path,dict(family='auxiliary CNN mean2',threshold=12,weights=weights,
            rule='fixed arithmetic mean of five internally fitted coefficient vectors; never fit on outer OOF predictions',
            frozen_before_test_inputs=True))
    frozen=json.loads(frozen_path.read_text())
    assert weights==frozen['weights'] and frozen['threshold']==12
    assert all(np.isfinite(v).all() and np.min(v)>=0 and np.max(v)<=1 for v in map(np.asarray,weights.values()))
    fitter=load_fitter()
    recipe=dict(family='auxiliary CNN, equal mean of two seeds, noise-regime calibration',seeds=list(SEEDS),
                seed_weights=[.5,.5],epochs=40,arm='aux',noise_threshold=12,production_weights=weights,
                production_weight_rule='Arithmetic mean of five B-only calibrated vectors; no global OOF fit',
                weight_source_sha256=r.digest(weights_source),source_weight_records=choices,
                calibration_frozen_sha256=r.digest(FINAL/'calibration-frozen.json'),
                training_ids=d['ids'].tolist(),normalization=data['meta'],test_used_for_fit=False,
                code_sha256=r.digest(__file__),fitter_sha256=r.digest(HERE/'full500_fitter.py'),
                plan_sha256=r.digest(HERE/'FINAL500.md'),protected=original)
    path=FINAL/'recipe.json'
    if path.exists():assert json.loads(path.read_text())==recipe
    else:r.save_json(path,recipe)
    output=FINAL/'models';output.mkdir(exist_ok=True);records=[];start=time.monotonic()
    for j,seed in enumerate(SEEDS):
        modeldir=output/f'aux-seed{j}'
        if not (modeldir/'training.json').exists():fitter.fit_one(data,'aux',seed,modeldir)
        metadata=json.loads((modeldir/'training.json').read_text())
        assert metadata['seed']==seed and metadata['epochs']==40 and metadata['meta']==data['meta']
        checkpoint=torch.load(modeldir/'model.pt',map_location='cpu',weights_only=True)
        model=fitter.ComplementCNN();model.load_state_dict(checkpoint['state_dict'])
        probe=np.arange(32)
        pred=fitter.predict(model,data['images'],data['context'],probe,'aux')*data['meta']['target_std']
        stored=np.load(modeldir/'predictions.npz');assert len(stored['validation_index'])==0
        error=float(np.max(np.abs(pred-stored['train_prediction'][probe])));assert error==0.
        records.append(dict(seed=seed,model_file=str((modeldir/'model.pt').relative_to(HERE)),
                            sha256=r.digest(modeldir/'model.pt'),training_probe_reload_max_error=error,
                            training_probe_is_not_validation=True))
        print('FINAL500 CNN COMPLETE',seed,'reload difference',error,flush=True)
    assert protected()==original
    r.save_json(FINAL/'fit-complete.json',dict(complete=True,models=records,seconds=time.monotonic()-start,
                                            n_training_images=500,test_inputs_read=False,validation_score_reported=False))


def forbid_training_inputs(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).absolute()
        if path==r.ROOT/'data/train.csv' or path.is_relative_to(r.ROOT/'data/train') or path==HERE.parent/'inputs.npz':
            raise RuntimeError('Training labels/pixels are not read during inference: '+str(path))


def infer():
    sys.addaudithook(forbid_training_inputs)
    recipe=json.loads((FINAL/'recipe.json').read_text());fit_record=json.loads((FINAL/'fit-complete.json').read_text())
    assert fit_record['complete'] and r.digest(__file__)==recipe['code_sha256']
    assert r.digest(FINAL/'calibration-frozen.json')==recipe['calibration_frozen_sha256']
    assert r.digest(HERE/'full500_fitter.py')==recipe['fitter_sha256']
    assert protected()==recipe['protected']
    used={};old=json.loads((HERE.parent/'reproduction/manifest.json').read_text())['input_sha256']
    old.update(json.loads((HERE.parent/'retrained-linear/manifest.json').read_text())['input_sha256'])
    def track(path):
        name=str(path.relative_to(r.ROOT));digest=r.digest(path)
        if name in old:assert digest==old[name],name
        used[name]=digest;return path
    order=[row['ID'] for row in csv.DictReader(track(r.ROOT/'data/sample_submission.csv').open())]
    assert len(order)==len(set(order))==1000 and all(i.startswith('TEST_') for i in order)
    def read_csv(tag):
        path=track(r.ROOT/f'artifacts/submission-candidate-{tag}.csv');rows=list(csv.DictReader(path.open()))
        assert [row['ID'] for row in rows]==order
        return np.array([float(row['hardness']) for row in rows])
    def read_npz(path,keys):
        with np.load(track(path),allow_pickle=True) as z:
            assert z['ids'].astype(str).tolist()==order
            return {key:z[key].copy() for key in keys}
    p073,p021=read_csv('073'),read_csv('021')
    ridge_mask=read_npz(r.EXP/'probabilistic-grains-900/ridge/final500/predictions.npz',['failed'])['failed']
    set_mask=read_npz(r.EXP/'probabilistic-grains-900/set_model/full500/predictions.npz',['failed'])['failed']
    failed=(ridge_mask|set_mask).astype(bool);assert np.array_equal(p073[failed],p021[failed])
    aug=read_npz(r.EXP/'push-1200/fin-d1234n12.npz',['noise','ridge'])
    pn=read_npz(r.H13/'fin-nb-partners.npz',['ridge_nb'])
    tn=read_npz(r.H13/'fin-nb-tab-cpu.npz',['nb_038','nb_039','nb_044'])
    features=read_npz(r.EXP/'push-800/feat-u1dmavgH2Z-anghcor-test.npz',['X','keys'])
    X=features['X'].astype(float);keys=features['keys'].astype(str).tolist()
    noise=np.nan_to_num(aug['noise'],nan=0);band=(noise>=12).astype(int)
    assert np.allclose(noise,np.nan_to_num(X[:,keys.index('noise')]*255,nan=0),atol=1e-4)
    tab_full=np.mean([(read_csv(slot)-.25*p021)/.75 for slot in ('038','039','044')],axis=0)
    tab_nb=np.mean([tn['nb_'+slot] for slot in ('038','039','044')],axis=0)
    tab=np.where(band,tab_full,tab_nb);ridge=np.where(band,aug['ridge'],pn['ridge_nb'])
    T=(p073-.16*p021)/.84;T[failed]=p073[failed]
    raw=np.c_[T,tab-ridge,noise,X[:,keys.index('u_ph')],
              np.log1p(np.maximum(np.nan_to_num(X[:,keys.index('n_obj')],nan=0),0)),X[:,keys.index('coverage')]]
    weights={key:np.asarray(value,float) for key,value in recipe['production_weights'].items()}
    active=(~failed)&((weights['addition'][band]>0)|(weights['replacement'][band]>0))
    active_indices=np.flatnonzero(active);images=[]
    for i in active_indices:
        path=track(r.ROOT/f'data/test/{order[i]}.png')
        with Image.open(path) as image:
            assert image.mode=='L' and image.size==(256,256)
            images.append(np.asarray(image,dtype=np.float32)/255.)
    images=np.stack(images)[:,None] if images else np.empty((0,1,256,256),np.float32)
    corrections=np.zeros((2,len(order)))
    for j,record in enumerate(fit_record['models']):
        path=HERE/record['model_file'];assert r.digest(path)==record['sha256']
        checkpoint=torch.load(path,map_location='cpu',weights_only=True);meta=checkpoint['normalization']
        assert meta==recipe['normalization'] and checkpoint['arm']=='aux' and checkpoint['seed']==SEEDS[j]
        model=e.c.ComplementCNN();model.load_state_dict(checkpoint['state_dict'])
        im=((images-meta['image_mean'])/meta['image_std']).astype(np.float32)
        clean=np.where(np.isfinite(raw),raw,np.asarray(meta['context_median']))
        context=((clean-np.asarray(meta['context_mean']))/np.asarray(meta['context_std'])).astype(np.float32)[active]
        pred=e.c.predict(model,im,context,np.arange(len(images)),'aux')*meta['target_std']
        assert np.isfinite(pred).all();corrections[j,active]=pred
    correction=corrections.mean(0);assert np.all(correction[failed]==0.)
    outputs={}
    for arm,base in [('addition',p073),('replacement',T)]:
        pred=base+weights[arm][band]*correction;pred[failed]=p073[failed]
        assert pred.shape==(1000,) and np.isfinite(pred).all() and np.array_equal(pred[failed],p073[failed])
        if arm=='addition':assert np.array_equal(pred[(band==0)&~failed],p073[(band==0)&~failed])
        path=FINAL/f'candidate-aux-noise-{arm}.csv'
        if path.exists():
            rows=list(csv.DictReader(path.open()));assert [row['ID'] for row in rows]==order
            assert np.array_equal(np.array([float(row['hardness']) for row in rows]),pred)
        else:
            with path.open('x',newline='') as f:
                writer=csv.writer(f);writer.writerow(['ID','hardness']);writer.writerows((i,repr(float(v))) for i,v in zip(order,pred))
        outputs[arm]=dict(file=str(path.relative_to(r.ROOT)),sha256=r.digest(path),n_rows=1000,
                          prediction_range=[float(pred.min()),float(pred.max())],
                          delta_rms_vs073=float(np.sqrt(np.mean((pred-p073)**2))),changed_rows=int(np.count_nonzero(pred!=p073)))
    np.savez_compressed(FINAL/'test-predictions.npz',ids=np.array(order),p073=p073,p021=p021,non021=T,
                        failed=failed,noise=noise,seed_corrections=corrections,mean_correction=correction,
                        context_raw=raw,active_indices=active_indices)
    assert protected()==recipe['protected']
    r.save_json(FINAL/'inference-verification.json',dict(complete=True,outputs=outputs,input_sha256=used,
        n_cnn_evaluated=int(active.sum()),fallback_count=int(failed.sum()),train_labels_or_pixels_read=False,
        fit_during_inference=False,weights_frozen_before_test=True,uploaded=False,production_recipe=recipe['production_weight_rule']))
    print(json.dumps(outputs,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('fit','infer'));args=parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):fit() if args.mode=='fit' else infer()
