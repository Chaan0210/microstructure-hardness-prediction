"""Frozen compact26, three-seed CNN and D4 robustness on development400.

New outer folds refit every component. Old OOF predictions are never reused.
Run this module in the project .venv; output is exclusively frozen-validation-021.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import argparse
import csv
import importlib.metadata
import json
import re
import sys
import time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():
    raise SystemExit('Use project .venv/bin/python -I frozen_multiseed.py')
sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
import torch
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
import frozen_validation as fv
from cnn_ensemble_inference import predict_views

OUTER_SEEDS=(20261031,20261101)
CNN_SEEDS=(20261003,20261023,20261043)
IDENTITY_ATOL=1e-4
SOURCE_FILES=('frozen_multiseed.py','frozen_validation.py','cnn_controls.py',
              'cnn_ensemble_inference.py','compact_regression.py',
              'candidate_features.py','grain_features.py','latent_features.py')
CAVEAT=(fv.CAVEAT+' The original CNN seed, compact panel, averaging and D4 hypotheses '
        'were explored on reused development data. These new partitions evaluate '
        'robustness, not independent confirmation. Primary recipe is fixed before training. '
        'Do not select the best split, initialization seed, or secondary endpoint.')


def source_hashes():
    return {name:fv.sha(_repo_path(ROOT, name)) for name in SOURCE_FILES}


def paired_summary(y, predictions, fid, split_seed):
    rng=np.random.default_rng(20261120+split_seed)
    draws=rng.integers(0,len(y),(5000,len(y)))
    scores={}
    references=('baseline','candidate003','mean3_identity','mean3_d4')
    reference_sq={name:(y-predictions[name])**2 for name in references}
    for name,pred in predictions.items():
        error=(y-pred)**2
        value={'oof_rmse':fv.rmse(y,pred),
               'fold_rmse':[fv.rmse(y[fid==k],pred[fid==k]) for k in range(5)],
               'bias':float((pred-y).mean()),'prediction_std':float(pred.std()),
               'paired_vs':{}}
        for ref in references:
            reference=predictions[ref]
            delta=np.sqrt(reference_sq[ref][draws].mean(axis=1))-np.sqrt(error[draws].mean(axis=1))
            fold_deltas=[fv.rmse(y[fid==k],reference[fid==k])-fv.rmse(y[fid==k],pred[fid==k]) for k in range(5)]
            value['paired_vs'][ref]={
                'improvement_rmse':fv.rmse(y,reference)-value['oof_rmse'],
                'fold_improvement_rmse':fold_deltas,
                'better_folds':sum(v>0 for v in fold_deltas),
                'conditional_paired_improvement_95':np.percentile(delta,[2.5,97.5]).tolist()}
        scores[name]=value
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary',required=True,choices=('mean3_identity','mean3_d4'),
                        help='Freeze the primary ensemble endpoint before any training')
    parser.add_argument('--extra-feature-run',help='Optional frozen development feature panel; secondary KRR comparison only')
    args=parser.parse_args()
    if args.extra_feature_run is not None and (not re.fullmatch(r'run-\d{3}',args.extra_feature_run) or args.extra_feature_run=='run-012'):
        parser.error('Extra panel must be a distinct run-NNN directory name')
    out=_repo_path(ROOT, 'artifacts/frozen-validation-021')
    if out.exists():
        raise FileExistsError('Output must be new; do not overwrite any prior run')
    panel_runs=['run-012']+([args.extra_feature_run] if args.extra_feature_run else [])
    ids,y,groups,Xbase,baseline_recipes,baseline_weights,panels,provenance=fv.read_inputs(panel_runs)
    compact=panels['run-012']
    if compact['X'].shape!=(400,26):
        raise ValueError('This protocol requires the frozen compact26 panel')
    torch.set_num_threads(2)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    source=source_hashes()
    plan={
        'primary':args.primary,'reference':'candidate003','feature_panel':'run-012',
        'feature_count':26,'outer_split_seeds':list(OUTER_SEEDS),'outer_folds':5,
        'extra_feature_run':args.extra_feature_run,
        'extra_feature_policy':'Optional CLI-frozen secondary panel; independent inner3 KRR and fixed baseline50/mean3D4_25/kernel25. No primary replacement or panel selection.',
        'cnn_base_seeds':list(CNN_SEEDS),'cnn_seed_rule':'base_seed + outer_fold_index',
        'cnn':{'architecture':'TinyCNN','image_size':128,'epochs':80,
               'average_last_epochs':5,'batch_size':32,'optimizer':'AdamW',
               'lr':.002,'weight_decay':.01,'cosine_T_max':80,'cosine_eta_min':.0001,
               'augmentation':'one uniform D4 transform per training batch',
               'normalization':'fit separately on each outer-training subset',
               'validation_checkpoint_selection':False},
        'kernel':{'inner_folds':3,'inner_seed_plus_fold':fv.INNER_SEED,
                  'bandwidths':list(fv.WIDTHS),'alphas':list(fv.ALPHAS),
                  'normalization':'refit within each inner training subset, then outer training'},
        'baseline_component_weights':baseline_weights,
        'ensemble_weights':{'baseline':.5,'compact_kernel':.25,'cnn':.25},
        'recipes':{
            'candidate003':'fresh baseline50 + compact kernel25 + first-seed identity CNN25',
            'mean3_identity':'same baseline/kernel + equal three-seed identity prediction mean25',
            'mean3_d4':'same baseline/kernel + equal three-seed/eight-D4-view prediction mean25'},
        'D4_view_order':[{'rotation_quarter_turns':k,'horizontal_flip':flip}
                         for k in range(4) for flip in (False,True)],
        'identity_reload_atol':IDENTITY_ATOL,'identity_reload_rtol':0.,
        'old_oof_prediction_reuse':False,'n_development':400,
        'decision_context':{
            'parent_selected_primary':'mean3_d4',
            'exploratory_original_fold_rmse':{'candidate003':13.705696,
                'mean3_identity':13.612514,'mean3_d4':13.585047},
            'limitation':'D4 alone improved only 2/5 original folds; new splits must assess this uncertainty.',
            'native_panel':'Excluded before this validation; native-feature models did not improve compact three-seed baseline.',
            'metric_values_source':'Parent-provided development summaries; no new data-independent evidence claimed'},
        'device':device,'source_sha256':source,
        'packages':{p:importlib.metadata.version(p) for p in
                    ('torch','numpy','scipy','Pillow','scikit-learn','joblib')},
        'caveat':CAVEAT}
    if args.primary!=plan['decision_context']['parent_selected_primary']:
        raise ValueError('Current frozen decision requires --primary mean3_d4')
    out.mkdir(parents=True,exist_ok=False)
    fv.write(out/'plan.json',plan)
    started=time.perf_counter()
    images,image_hashes=fv.load_development_images(ids)
    provenance['source_sha256']=source
    provenance['development_image_sha256']=image_hashes
    fv.write(out/'input-provenance.json',provenance)
    seed_reports=[]
    with (out/'progress.jsonl').open('x',encoding='utf-8') as progress:
        for split_seed in OUTER_SEEDS:
            directory=out/f'seed-{split_seed}'
            directory.mkdir()
            folds=list(GroupKFold(n_splits=5,shuffle=True,random_state=split_seed).split(Xbase,y,groups))
            fid=np.full(len(y),-1,dtype=int)
            for k,(_,va) in enumerate(folds):
                fid[va]=k
            if np.any(fid<0):
                raise ValueError('Incomplete outer fold assignments')
            fv.write(directory/'split_manifest.json',{
                'seed':split_seed,'development_folds':dict(zip(ids,fid.tolist())),
                'development_groups':dict(zip(ids,groups.tolist())),
                'ordered_development_ids':ids})
            prediction_names=['baseline','compact_kernel',
                *[f'cnn_seed{s}_identity' for s in CNN_SEEDS],
                *[f'cnn_seed{s}_d4' for s in CNN_SEEDS],
                'cnn_mean3_identity','cnn_mean3_d4','candidate003','mean3_identity','mean3_d4']
            if args.extra_feature_run:
                prediction_names.extend(['extra_kernel','extra_mean3_d4'])
            predictions={n:np.full(len(y),np.nan) for n in prediction_names}
            fold_records=[]
            for fold,(tr,va) in enumerate(folds):
                if set(groups[tr])&set(groups[va]):
                    raise ValueError('Outer training/validation group overlap')
                fold_start=time.perf_counter()
                checkpoint_dir=directory/f'fold-{fold}'
                checkpoint_dir.mkdir()
                t=time.perf_counter()
                fitted={n:clone(m).fit(Xbase[tr],y[tr]) for n,m in baseline_recipes.items()}
                baseline=sum(baseline_weights[n]*m.predict(Xbase[va]) for n,m in fitted.items())
                predictions['baseline'][va]=baseline
                joblib.dump({'models':fitted,'weights':baseline_weights,
                    'feature_names':provenance['baseline_feature_names'],
                    'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va]},
                    _repo_io(checkpoint_dir/'baseline.joblib'))
                baseline_seconds=time.perf_counter()-t
                del fitted
                t=time.perf_counter()
                kernel,kernel_model,kernel_selection=fv.kernel_fold(compact['X'],y,groups,
                    compact['blocks'],tr,va,fold)
                predictions['compact_kernel'][va]=kernel
                joblib.dump({'model':kernel_model,'feature_names':compact['names'],
                    'blocks':compact['blocks'],'train_ids':[ids[i] for i in tr],
                    'validation_ids':[ids[i] for i in va]},_repo_io(checkpoint_dir/'kernel.joblib'))
                kernel_seconds=time.perf_counter()-t
                del kernel_model
                extra_record=None
                if args.extra_feature_run:
                    t=time.perf_counter()
                    panel=panels[args.extra_feature_run]
                    extra_pred,extra_model,extra_selection=fv.kernel_fold(panel['X'],y,groups,panel['blocks'],tr,va,fold)
                    predictions['extra_kernel'][va]=extra_pred
                    joblib.dump({'model':extra_model,'feature_names':panel['names'],'blocks':panel['blocks'],
                        'panel':args.extra_feature_run,'train_ids':[ids[i] for i in tr],
                        'validation_ids':[ids[i] for i in va]},_repo_io(checkpoint_dir/'extra-kernel.joblib'))
                    extra_record={'panel':args.extra_feature_run,'selection':extra_selection,
                        'seconds':time.perf_counter()-t,'feature_count':len(panel['names'])}
                    del extra_model
                cnn_records=[]
                all_views=[]
                original_seed=fv.CNN_SEED
                try:
                    for base_seed in CNN_SEEDS:
                        fv.CNN_SEED=base_seed
                        checkpoint=checkpoint_dir/f'cnn-seed{base_seed}.pt'
                        t=time.perf_counter()
                        identity,train_summary=fv.cnn_fold(images,y,tr,va,ids,fold,device,checkpoint)
                        training_seconds=time.perf_counter()-t
                        t=time.perf_counter()
                        views=predict_views(images[va],checkpoint,device=device)
                        if views.shape!=(8,len(va)):
                            raise ValueError('Invalid D4 view shape')
                        np.testing.assert_allclose(views[0],identity,rtol=0,atol=IDENTITY_ATOL,
                            err_msg='Reloaded identity differs from fresh cnn_fold output')
                        # Preserve exact cnn_fold identity for the reference recipe.
                        predictions[f'cnn_seed{base_seed}_identity'][va]=identity
                        predictions[f'cnn_seed{base_seed}_d4'][va]=views.mean(axis=0)
                        all_views.append(views)
                        cnn_records.append({'base_seed':base_seed,'effective_seed':base_seed+fold,
                            'training_seconds':training_seconds,'D4_prediction_seconds':time.perf_counter()-t,
                            'identity_reload_max_abs_difference':float(np.max(np.abs(views[0]-identity))),
                            'D4_validation_rmse':fv.rmse(y[va],views.mean(axis=0)),**train_summary})
                        print(json.dumps({'split_seed':split_seed,'fold':fold,'cnn_base_seed':base_seed,
                            'identity_rmse':train_summary['validation_rmse'],
                            'D4_rmse':cnn_records[-1]['D4_validation_rmse']}),flush=True)
                finally:
                    fv.CNN_SEED=original_seed
                np.savez_compressed(_repo_io(checkpoint_dir/'validation_views.npz'),
                    ids=np.array([ids[i] for i in va]),base_seeds=np.array(CNN_SEEDS),
                    predictions=np.stack(all_views))
                mean_identity=np.mean([predictions[f'cnn_seed{s}_identity'][va] for s in CNN_SEEDS],axis=0)
                mean_d4=np.mean([predictions[f'cnn_seed{s}_d4'][va] for s in CNN_SEEDS],axis=0)
                predictions['cnn_mean3_identity'][va]=mean_identity
                predictions['cnn_mean3_d4'][va]=mean_d4
                for name,cnn in [('candidate003',predictions[f'cnn_seed{CNN_SEEDS[0]}_identity'][va]),
                                 ('mean3_identity',mean_identity),('mean3_d4',mean_d4)]:
                    predictions[name][va]=.5*baseline+.25*kernel+.25*cnn
                if args.extra_feature_run:
                    predictions['extra_mean3_d4'][va]=.5*baseline+.25*predictions['extra_kernel'][va]+.25*mean_d4
                record={'split_seed':split_seed,'fold':fold,'n_train':len(tr),'n_validation':len(va),
                    'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va],
                    'baseline_seconds':baseline_seconds,'kernel_seconds':kernel_seconds,
                    'kernel_selection':kernel_selection,'extra_kernel':extra_record,'cnn':cnn_records,
                    'validation_rmse':{n:fv.rmse(y[va],p[va]) for n,p in predictions.items()},
                    'elapsed_seconds':time.perf_counter()-fold_start,
                    'checkpoint_sha256':{p.name:fv.sha(p) for p in checkpoint_dir.iterdir()}}
                fv.write(directory/f'fold-{fold}-report.json',record)
                fold_records.append(record)
                event={k:record[k] for k in ('split_seed','fold','validation_rmse','elapsed_seconds')}
                progress.write(json.dumps(event,allow_nan=False)+'\n')
                progress.flush()
                print(json.dumps(event),flush=True)
            scores=paired_summary(y,predictions,fid,split_seed)
            with (directory/'development_oof.csv').open('x',newline='',encoding='utf-8') as f:
                writer=csv.writer(f)
                writer.writerow(['ID','hardness','fold',*predictions])
                writer.writerows([ident,y[i],fid[i],*[p[i] for p in predictions.values()]]
                                 for i,ident in enumerate(ids))
            report={'split_seed':split_seed,'primary':args.primary,'scores':scores,
                    'folds':fold_records,'target_std':float(y.std()),'caveat':CAVEAT}
            fv.write(directory/'report.json',report)
            seed_reports.append({'split_seed':split_seed,'scores':scores,
                'primary_improvement_vs_candidate003':scores[args.primary]['paired_vs']['candidate003']})
    if source_hashes()!=source:
        raise RuntimeError('Source changed during validation; partial outputs require review')
    fv.write(out/'report.json',{**plan,'seeds':seed_reports,
        'elapsed_seconds':time.perf_counter()-started,
        'summary_policy':'Report both predeclared partitions; no best-split selection or pooled800 confidence interval',
        'primary_decision':'Results for review; no automatic model promotion or artifact replacement'})
    print(json.dumps({'output':str(out.relative_to(ROOT)),'primary':args.primary,
        'seed_summaries':[{'split_seed':r['split_seed'],
            'primary_vs_candidate003':r['primary_improvement_vs_candidate003']} for r in seed_reports]}),flush=True)


if __name__=='__main__':
    main()
