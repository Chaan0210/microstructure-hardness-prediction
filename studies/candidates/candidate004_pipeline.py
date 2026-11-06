"""Frozen candidate004: baseline50 + compact kernel25 + three-seed D4 CNN25.

Only final fit reads all train500, after completed development reports and a root
panel decision are verified and a local decision is saved. Prediction never fits.
No upload or replacement of candidate003 is performed.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='2'
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():
    raise SystemExit('Use project .venv/bin/python -I candidate004_pipeline.py')
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import torch
from PIL import Image
import candidate_pipeline as cp
from candidate_features import extract_candidate
from denoised_grain_features import extract_denoised
from compact_regression import BalancedKernel
from cnn_ensemble_inference import predict_views

ORIGINAL=_repo_path(ROOT, 'artifacts/final-model-017')
ROOT_DECISION=_repo_path(ROOT, 'reports/research-decision-024.json')
CNN_SEEDS=(20261003,20261023,20261043)
SOURCE_FILES=('candidate004_pipeline.py','candidate_pipeline.py','candidate_features.py',
              'denoised_grain_features.py','cnn_ensemble_inference.py','latent_features.py',
              'grain_features.py','compact_regression.py','cnn_controls.py')


def sources():
    return {n:cp.sha(_repo_path(ROOT, n)) for n in SOURCE_FILES}


def protected():
    result=cp.protected()
    files=[_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'),
           *[_repo_path(ORIGINAL, n) for n in ('baseline.joblib','kernel.joblib','cnn.pt','report.json','decision.json')]]
    result.update({str(p.relative_to(ROOT)):cp.sha(p) for p in files})
    if cp.sha(_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'))!='5ab625d8aa5987ba9f46e3e52b5690cb39e8514e5ae557966522787ccc98c693':
        raise ValueError('Known candidate003 CSV hash mismatch')
    report=json.loads((_repo_path(ORIGINAL, 'report.json')).read_text())
    for name,digest in report['model_sha256'].items():
        if cp.sha(_repo_path(ORIGINAL, name))!=digest:
            raise ValueError('Candidate003 model integrity mismatch: '+name)
    if cp.sha(_repo_path(ORIGINAL, 'decision.json'))!=report['decision_sha256']:
        raise ValueError('Candidate003 decision integrity mismatch')
    if cp.packages()!=report['packages'] or cp.sources(report['panel'])!=report['source_sha256']:
        raise ValueError('Candidate003 source or dependency versions changed')
    return result,report


def copy_exclusive(source,destination):
    with source.open('rb') as src,destination.open('xb') as dst:
        shutil.copyfileobj(src,dst)
    if cp.sha(source)!=cp.sha(destination):
        raise ValueError('Copied artifact integrity mismatch')


def dump_exclusive(value,path):
    with path.open('xb') as f:
        joblib.dump(value,_repo_io(f))


def completed_decision(panel):
    root=json.loads(ROOT_DECISION.read_text())
    selected=root.get('panel',root.get('selected_panel'))
    if selected!=panel:
        raise ValueError('CLI panel must equal root decision panel/selected_panel')
    if 'cnn_seeds' in root and root['cnn_seeds']!=list(CNN_SEEDS):
        raise ValueError('Root decision CNN seeds differ from this frozen implementation')
    if 'cnn_aggregation' in root and root['cnn_aggregation']!='mean3_d4':
        raise ValueError('Root decision must use mean3_d4 aggregation')
    paths=(_repo_path(ROOT, 'artifacts/frozen-validation-021/report.json'),
           _repo_path(ROOT, 'artifacts/denoised-confirmation-023/report.json'))
    first,second=[json.loads(p.read_text()) for p in paths]
    expected={20261031,20261101}
    if (len(first.get('seeds',[]))!=2 or {r['split_seed'] for r in first['seeds']}!=expected
            or first.get('primary')!='mean3_d4' or 'elapsed_seconds' not in first):
        raise ValueError('Completed frozen-validation-021 is required')
    if (len(second.get('seeds',[]))!=2 or {r['seed'] for r in second['seeds']}!=expected
            or 'elapsed_seconds' not in second):
        raise ValueError('Completed denoised-confirmation-023 is required')
    if second.get('parent_validation_report_sha256')!=cp.sha(paths[0]):
        raise ValueError('Denoised confirmation does not match frozen021 report')
    for record in first['seeds']:
        if 'mean3_d4' not in record['scores']:
            raise ValueError('Missing primary validation result')
    for record in second['seeds']:
        if 'denoised_candidate' not in record['scores']:
            raise ValueError('Missing secondary validation result')
    return {'root_decision':root,'root_decision_sha256':cp.sha(ROOT_DECISION),
            'validation_report_sha256':{str(p.relative_to(ROOT)):cp.sha(p) for p in paths},
            'interpretation':'Repeated development400 robustness; not independent validation'}


def extract_one(ident,split,panel):
    path=_repo_path(ROOT, 'data', split, ident+'.png')
    if not path.resolve().is_relative_to((_repo_path(ROOT, 'data', split)).resolve()):
        raise ValueError('Image symlink escapes selected project data directory')
    with Image.open(_repo_io(path)) as im:
        if im.mode!='L' or im.size!=(256,256):
            raise ValueError('Invalid image schema: '+ident)
        raw=np.asarray(im,dtype=np.uint8).copy()
        image=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)/255.
    x,names=extract_candidate(raw,'run-012')
    if panel=='run-022':
        more,more_names,_=extract_denoised(raw)
        x=np.concatenate((x,more));names=names+more_names
    elif panel!='run-012':
        raise ValueError('Unsupported panel')
    return x,names,image,cp.sha(path)


def extract_all(ids,split,panel,names):
    records=[]
    for first in range(0,len(ids),40):
        records.extend(joblib.Parallel(n_jobs=2)(joblib.delayed(extract_one)(s,split,panel)
                                               for s in ids[first:first+40]))
        print(f'{split}: extracted {min(first+40,len(ids))}/{len(ids)}',flush=True)
    expected=57 if panel=='run-022' else 26
    if len(names)!=expected or any(r[1]!=names for r in records):
        raise ValueError('Frozen candidate feature schema mismatch')
    X=np.stack([r[0] for r in records]);images=np.stack([r[2] for r in records])[:,None]
    if X.shape!=(len(ids),expected) or not np.isfinite(X).all() or not np.isfinite(images).all():
        raise ValueError('Invalid extracted inputs')
    return X,images,dict(zip(ids,[r[3] for r in records]))


def assert_normalization(checkpoint,images,y,ids,seed):
    saved=torch.load(_repo_io(checkpoint),map_location='cpu',weights_only=True)
    if saved['train_ids']!=ids or saved.get('seed')!=seed or saved.get('epochs')!=80:
        raise ValueError('CNN training IDs, seed or endpoint mismatch')
    expected=[float(images.mean()),float(images.std()),float(y.mean()),float(y.std())]
    actual=[saved['normalization'][k] for k in ('x_mean','x_std','y_mean','y_std')]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10,
                               err_msg='CNN fulltrain normalization mismatch')
    return saved['normalization']


def fit_cnn(X,y,ids,output,device,seed):
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    mu,sd,ym,ys=float(X.mean()),float(X.std()),float(y.mean()),float(y.std())
    if sd<=0 or ys<=0:
        raise ValueError('Degenerate CNN normalization')
    Xt,yt=(X-mu)/max(sd,1e-6),(y-ym)/ys
    model=cp.engine.TinyCNN().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=.0001)
    avg,curves=None,[]
    for epoch in range(80):
        model.train();order=rng.permutation(len(y));loss_sum=0.
        for first in range(0,len(y),32):
            ix=order[first:first+32]
            xb=np.rot90(Xt[ix],int(rng.integers(4)),axes=(2,3))
            if rng.random()<.5:xb=xb[:,:,:,::-1]
            xb=torch.from_numpy(np.ascontiguousarray(xb)).to(device)
            target=torch.tensor(yt[ix],dtype=torch.float32,device=device)
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(xb),target)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite CNN training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            loss_sum+=float(loss.detach().cpu())*len(ix)
        schedule.step()
        curves.append({'epoch':epoch+1,'augmented_train_rmse':float(np.sqrt(loss_sum/len(y))*ys),
                       'lr_after_step':schedule.get_last_lr()[0]})
        if epoch>=75:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if avg is None:avg={k:v/5 for k,v in state.items()}
            else:
                for k in avg:avg[k]+=state[k]/5
        if (epoch+1)%10==0:print(json.dumps({'seed':seed,**curves[-1]}),flush=True)
    cp.engine.sync(device)
    with output.open('xb') as f:
        torch.save({'state_dict':avg,'normalization':{'x_mean':mu,'x_std':sd,'y_mean':ym,'y_std':ys},
                    'train_ids':ids,'architecture':'TinyCNN','image_size':128,'seed':seed,
                    'epochs':80,'average_last_epochs':5},_repo_io(f))
    del model,opt,avg
    if device=='mps':torch.mps.empty_cache()
    return curves


def fit(args):
    out=cp.local(args.output,artifacts=True)
    if out.exists():raise FileExistsError('Final model output must be new')
    evidence=completed_decision(args.panel)
    snapshot,original_report=protected()
    selected,names,blocks,dev_ids,dev_y,dev_X,selection=cp.choose_kernel(args.panel)
    if len(names)!=(57 if args.panel=='run-022' else 26):
        raise ValueError('Unexpected selected panel width')
    decision={'panel':args.panel,'evidence':evidence,'kernel':selected,'kernel_selection':selection,
        'feature_names':names,'blocks':blocks,'cnn_seeds':list(CNN_SEEDS),'cnn_aggregation':'mean3_d4',
        'ensemble_weights':{'baseline':.5,'kernel':.25,'cnn_mean3_d4':.25},
        'cnn':{'architecture':'TinyCNN','image_size':128,'epochs':80,'batch_size':32,
               'average_last_epochs':5,'optimizer':'AdamW','lr':.002,'weight_decay':.01,
               'cosine_T_max':80,'cosine_eta_min':.0001,'gradient_clip_norm':5.,
               'training_augmentation':'batch D4','inference':'equal prediction mean of 3 seeds x 8 D4 views'},
        'reuse':{'baseline':'Exact candidate003 baseline bytes','cnn_seed20261003':'Exact candidate003 CNN bytes',
                'kernel':'Development-only parameter selection followed by fulltrain500 fit'},
        'source_sha256':sources(),'packages':cp.packages(),'protected_artifacts':snapshot,
        'scope':'Recipe locked before all train500 labels; no holdout scoring or test fitting',
        'license_references':{'project_code':'Original local implementation; no pretrained weights or downloaded model code',
             'dependencies':'Same installed NumPy/SciPy/Pillow/scikit-learn/scikit-image/joblib/PyTorch distributions as candidate003; see distribution METADATA and license files'}}
    out.mkdir(parents=True,exist_ok=False)
    cp.write(out/'decision.json',decision)
    # No full training labels/images are accessed before the decision above.
    started=time.perf_counter()
    ids,y=cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True)
    if cp.sha(_repo_path(ROOT, 'data/train.csv'))!=original_report['train_csv_sha256'] or ids!=original_report['training_id_order']:
        raise ValueError('Training CSV or order changed since candidate003')
    index={s:i for i,s in enumerate(ids)}
    if not set(dev_ids).issubset(index):raise ValueError('Missing development IDs')
    np.testing.assert_array_equal(y[[index[s] for s in dev_ids]],dev_y)
    X,images,image_hashes=extract_all(ids,'train',args.panel,names)
    if image_hashes!=original_report['training_image_sha256']:
        raise ValueError('Training image bytes changed since candidate003')
    np.testing.assert_allclose(X[[index[s] for s in dev_ids]],dev_X,rtol=1e-10,atol=1e-12,
                               err_msg='Fresh features differ from frozen development panel')
    copy_exclusive(_repo_path(ORIGINAL, 'baseline.joblib'),out/'baseline.joblib')
    copy_exclusive(_repo_path(ORIGINAL, 'cnn.pt'),out/'cnn-seed20261003.pt')
    normalizations={str(CNN_SEEDS[0]):assert_normalization(out/'cnn-seed20261003.pt',images,y,ids,CNN_SEEDS[0])}
    kernel=BalancedKernel(blocks,selected['bandwidth'],selected['alpha']).fit(X,y)
    dump_exclusive({'model':kernel,'feature_names':names,'blocks':blocks,'train_ids':ids},out/'kernel.joblib')
    torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu'
    for seed in CNN_SEEDS[1:]:
        checkpoint=out/f'cnn-seed{seed}.pt'
        curves=fit_cnn(images,y,ids,checkpoint,device,seed)
        cp.write(out/f'training-curve-seed{seed}.json',curves)
        normalizations[str(seed)]=assert_normalization(checkpoint,images,y,ids,seed)
    if sources()!=decision['source_sha256']:raise RuntimeError('Source changed during fit')
    if cp.sha(ROOT_DECISION)!=evidence['root_decision_sha256']:raise RuntimeError('Root decision changed during fit')
    for name,digest in evidence['validation_report_sha256'].items():
        if cp.sha(_repo_path(ROOT, name))!=digest:raise RuntimeError('Validation report changed during fit')
    cp.assert_protected(snapshot)
    model_names=['baseline.joblib','kernel.joblib',*[f'cnn-seed{s}.pt' for s in CNN_SEEDS]]
    report={'panel':args.panel,'n_train':500,'decision_sha256':cp.sha(out/'decision.json'),
        'source_sha256':decision['source_sha256'],'packages':decision['packages'],
        'model_sha256':{n:cp.sha(out/n) for n in model_names},'feature_names':names,'blocks':blocks,
        'train_csv_sha256':cp.sha(_repo_path(ROOT, 'data/train.csv')),'training_image_sha256':image_hashes,
        'training_id_order':ids,'cnn_seeds':list(CNN_SEEDS),'cnn_normalizations':normalizations,
        'cnn_aggregation':'mean3_d4','device':device,'elapsed_seconds':time.perf_counter()-started,
        'protected_artifacts':snapshot,'test_used':False,'holdout_scored':False,
        'validation':'Development-only confirmation; no fulltrain validation scoring',
        'reproducibility':'Seeded fitting and CPU-portable saved tensors; MPS bitwise repeatability not guaranteed'}
    cp.write(out/'report.json',report)
    print(json.dumps({'model':str(out.relative_to(ROOT)),'panel':args.panel,'seconds':report['elapsed_seconds']}),flush=True)


def predict(args):
    directory=cp.local(args.model,artifacts=True);output=cp.local(args.output,artifacts=True)
    components=output.with_suffix('.components.csv');manifest=output.with_suffix('.manifest.json')
    viewfile=output.with_suffix('.cnn-views.npz')
    if any(p.exists() for p in (output,components,manifest,viewfile)):
        raise FileExistsError('Every prediction output must be new')
    snapshot,_=protected()
    report=json.loads((directory/'report.json').read_text())
    if cp.sha(directory/'decision.json')!=report['decision_sha256']:raise ValueError('Decision integrity mismatch')
    decision=json.loads((directory/'decision.json').read_text())
    if report['source_sha256']!=sources() or report['packages']!=cp.packages():
        raise ValueError('Source/dependencies changed since fit')
    if snapshot!=report['protected_artifacts']:raise ValueError('Protected originals changed since fit')
    if report['cnn_seeds']!=list(CNN_SEEDS) or report['cnn_aggregation']!='mean3_d4':
        raise ValueError('Unexpected saved CNN recipe')
    for name,digest in report['model_sha256'].items():
        if cp.sha(directory/name)!=digest:raise ValueError('Model integrity mismatch: '+name)
    # Locally produced joblib artifacts are loaded only after integrity checks.
    baseline=joblib.load(_repo_io(directory/'baseline.joblib'));kernel=joblib.load(_repo_io(directory/'kernel.joblib'))
    sample=_repo_path(ROOT, 'data/sample_submission.csv');ids,_=cp.read_csv(sample,'TEST',1000,False)
    cachepath=_repo_path(ROOT, 'artifacts/test-prediction-features.npz')
    with np.load(_repo_io(cachepath),allow_pickle=False) as cache:
        if cache['ids'].tolist()!=ids or cache['names'].tolist()!=baseline['feature_names']:
            raise ValueError('Baseline cached ID order or schema mismatch')
        Xbase=cache['X'];cached_hashes=cache['hashes'].tolist()
    if Xbase.shape!=(1000,len(baseline['feature_names'])) or not np.isfinite(Xbase).all():
        raise ValueError('Invalid cached baseline features')
    if kernel['feature_names']!=report['feature_names'] or kernel['blocks']!=report['blocks']:
        raise ValueError('Kernel schema mismatch')
    started=time.perf_counter()
    X,images,image_hashes=extract_all(ids,'test',report['panel'],report['feature_names'])
    if cached_hashes!=[image_hashes[s] for s in ids]:raise ValueError('Baseline test image hashes differ')
    base=sum(baseline['weights'][n]*m.predict(Xbase) for n,m in baseline['models'].items())
    kp=kernel['model'].predict(X)
    torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu'
    views=[];values={'baseline':base,'kernel':kp}
    for seed in CNN_SEEDS:
        v=predict_views(images,directory/f'cnn-seed{seed}.pt',device=device)
        views.append(v);values[f'cnn_seed{seed}_identity']=v[0];values[f'cnn_seed{seed}_d4']=v.mean(axis=0)
    stacked=np.stack(views)
    if stacked.shape!=(3,8,1000):raise ValueError('Unexpected seed/view dimensions')
    values['cnn_mean3_identity']=stacked[:,0,:].mean(axis=0)
    values['cnn_mean3_d4']=stacked.mean(axis=(0,1))
    values['ensemble']=.5*base+.25*kp+.25*values['cnn_mean3_d4']
    if any(p.shape!=(1000,) or not np.isfinite(p).all() for p in values.values()):
        raise ValueError('Invalid component predictions')
    if sources()!=report['source_sha256']:raise RuntimeError('Source changed during prediction')
    cp.assert_protected(snapshot)
    output.parent.mkdir(parents=True,exist_ok=True)
    with viewfile.open('xb') as f:
        np.savez_compressed(_repo_io(f),ids=np.array(ids),seeds=np.array(CNN_SEEDS),predictions=stacked)
    with components.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['ID',*values])
        w.writerows([ident,*[p[i] for p in values.values()]] for i,ident in enumerate(ids))
    with output.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,values['ensemble']))
    cp.write(manifest,{'model':str(directory.relative_to(ROOT)),'panel':report['panel'],
        'model_report_sha256':cp.sha(directory/'report.json'),'decision_sha256':report['decision_sha256'],
        'model_sha256':report['model_sha256'],'source_sha256':report['source_sha256'],
        'packages':report['packages'],'sample_submission_sha256':cp.sha(sample),
        'baseline_test_cache_sha256':cp.sha(cachepath),'image_sha256':image_hashes,
        'submission_sha256':cp.sha(output),'components_sha256':cp.sha(components),'cnn_views_sha256':cp.sha(viewfile),
        'rows':1000,'exact_sample_order':True,'prediction_only':True,'clipping':False,
        'ensemble_weights':decision['ensemble_weights'],'cnn_seeds':list(CNN_SEEDS),
        'cnn_view_order':[{'rotation_quarter_turns':k,'horizontal_flip':flip}
                          for k in range(4) for flip in (False,True)],
        'cnn_aggregation':'equal prediction mean of 3 seeds x 8 views',
        'device':device,'elapsed_seconds':time.perf_counter()-started,'protected_artifacts':snapshot,
        'component_summary':{n:{'min':float(p.min()),'max':float(p.max()),'mean':float(p.mean()),
                                'std':float(p.std())} for n,p in values.items()}})
    cp.assert_protected(snapshot)
    print(f'Prepared {output.relative_to(ROOT)}; no upload performed.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('fit');p.add_argument('--panel',required=True,choices=('run-012','run-022'))
    p.add_argument('--output',default='artifacts/final-model-024');p.set_defaults(func=fit)
    p=sub.add_parser('predict');p.add_argument('--model',default='artifacts/final-model-024')
    p.add_argument('--output',default='artifacts/submission-candidate-004.csv');p.set_defaults(func=predict)
    args=parser.parse_args();args.func(args)


if __name__=='__main__':
    main()
