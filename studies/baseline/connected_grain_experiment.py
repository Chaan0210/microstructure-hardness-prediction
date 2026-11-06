"""One scoped connected-boundary hypothesis: QA first, then development-only CV."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import argparse,csv,hashlib,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():
    raise SystemExit('Use project .venv/bin/python -I')
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from connected_grain_features import extract_connected,segment_connected,FEATURE_NAMES,BLOCKS,PARAMETERS
OUT=_repo_path(ROOT, 'artifacts/run-028')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def manifest():
    sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text())
    ids=sorted(sp['development_folds']);assert len(ids)==400
    return sp,ids
def read_image(ident,allow):
    if ident not in allow:raise ValueError('Image outside development allowlist')
    p=_repo_path(ROOT, 'data/train', ident+'.png')
    if not p.resolve().is_relative_to((_repo_path(ROOT, 'data/train')).resolve()):raise ValueError('Image path escape')
    with Image.open(_repo_io(p)) as im:
        if im.mode!='L' or im.size!=(256,256):raise ValueError('Bad image schema')
        return np.asarray(im,dtype=np.uint8).copy()
def records(run):
    _,ids=manifest()
    with (_repo_path(ROOT, 'artifacts', run, 'development_oof.csv')).open() as f:rr=list(csv.DictReader(f))
    if len(rr)!=400 or {r['ID'] for r in rr}!=set(ids):raise ValueError('OOF not development400')
    return {r['ID']:r for r in rr}
def panel():
    sp,ids=manifest();r=json.loads((_repo_path(ROOT, 'artifacts/run-012/report.json')).read_text())
    with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as a:
        if len(a['ids'])!=400 or set(a['ids'])!=set(ids):raise ValueError('Non-development cache')
        lookup={s:i for i,s in enumerate(a['ids'])};X=a['X'][[lookup[s] for s in ids]]
    assert X.shape==(400,26) and np.isfinite(X).all()
    return sp,ids,X,r['features'],r['blocks']
def qa():
    sp,ids=manifest();selection=json.loads((_repo_path(OUT, 'qa-selection.json')).read_text())['selected']
    assert len(selection)==16 and len({r['ID'] for r in selection})==16 and all(r['ID'] in ids for r in selection)
    rng=np.random.default_rng(20261007);yy,xx=np.indices((256,256));grid=np.where((xx%32<3)|(yy%32<3),40.,180.)
    fixtures={'constant':np.full((256,256),127,np.uint8),'grid':grid.astype(np.uint8),'noisy_grid':np.clip(grid+rng.normal(0,8,grid.shape),0,255).astype(np.uint8),'stripes':np.where(xx%24<3,30,190).astype(np.uint8)}
    checks=[]
    for name,img in fixtures.items():
        v,n,b=extract_connected(img);assert len(set(n))==len(n)==22 and n==list(FEATURE_NAMES) and b==list(BLOCKS)
        repeat,_,_=extract_connected(img);np.testing.assert_array_equal(v,repeat)
        for k in range(4):
            for z in (np.rot90(img,k),np.fliplr(np.rot90(img,k))):
                w,_,_=extract_connected(z);np.testing.assert_allclose(v,w,rtol=1e-10,atol=1e-10)
        checks.append({'fixture':name,'values':dict(zip(n,v.tolist()))})
    for bad in (np.zeros((128,128),np.uint8),np.zeros((256,256),float)):
        try:extract_connected(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid image accepted')
    output=[]
    for noise_group in range(4):
        sheet=Image.new('RGB',(3*264,4*286),(255,255,255));draw=ImageDraw.Draw(sheet)
        for row,e in enumerate(selection[noise_group*4:(noise_group+1)*4]):
            img=read_image(e['ID'],ids);s=segment_connected(img);v,n,b=extract_connected(img)
            raw=np.repeat(s['image'][:,:,None],3,axis=2)
            overlay=raw.copy();overlay[s['network']]=(255,50,30);overlay[s['spots']]=(40,120,255)
            lab=s['basins'];colors=np.stack(((lab*47+31)%230,(lab*79+67)%230,(lab*103+113)%230),axis=2).astype(np.uint8)
            colors[lab==0]=(0,0,0);colors=(.3*raw+.7*colors).astype(np.uint8)
            for col,a in enumerate((raw,overlay,colors)):
                sheet.paste(Image.fromarray(a),(col*264,row*286+27))
                label=e['ID'] if col==0 else ('red=network blue=spots' if col==1 else f'closed={v[1]:.0f} area={v[2]:.2f} max={v[4]:.2f}')
                draw.text((col*264+2,row*286+5),label,fill=(0,0,0))
            output.append({'ID':e['ID'],'values':dict(zip(n,v.tolist())),'image_sha256':sha(_repo_path(ROOT, 'data/train', e['ID']+'.png'))})
        sheet.save(_repo_path(OUT, f'qa-network-noise{noise_group+1}.png'))
    write(_repo_path(OUT, 'qa-engineering.json'),{'source_sha256':sha(_repo_path(ROOT, 'connected_grain_features.py')),'fixtures':checks,'development_images':output,'D4_checks':32,'labels_read':False,'fixed_parameters':PARAMETERS})
    print(json.dumps({'qa_complete':True,'images':16,'features':22,'source_sha256':sha(_repo_path(ROOT, 'connected_grain_features.py'))}),flush=True)
def fit():
    import frozen_validation as fv
    from threadpoolctl import threadpool_limits
    # Imported historical modules set thread environment variables to2; enforce
    # one thread after all imports and keep the runtime limiter active below.
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
    fv.torch.set_num_threads(1)
    with threadpool_limits(limits=1):fit_limited(fv)
def fit_limited(fv):
    start=time.perf_counter();sp,ids,old,old_names,old_blocks=panel()
    approval=json.loads((_repo_path(OUT, 'qa-review.json')).read_text())
    if not approval['proceed_to_development_cv'] or approval['source_sha256']!=sha(_repo_path(ROOT, 'connected_grain_features.py')):raise ValueError('Missing matching-source QA decision')
    r12=records('run-012');r9=records('run-009')
    y=np.array([float(r12[s]['hardness']) for s in ids]);assert np.array_equal(y,np.array([float(r9[s]['hardness']) for s in ids]))
    baseline=np.array([float(r12[s]['baseline']) for s in ids]);cnn=np.array([float(r9[s]['cnn']) for s in ids]);prior_kernel=np.array([float(r12[s]['nested_kernel']) for s in ids]);candidate003=np.array([float(r12[s]['nested_kernel_cnnblend']) for s in ids])
    np.testing.assert_allclose(candidate003,.5*baseline+.25*cnn+.25*prior_kernel,rtol=0,atol=1e-12)
    groups=np.array([sp['group_by_id'][s] for s in ids]);fid=np.array([sp['development_folds'][s] for s in ids]);names=old_names+list(FEATURE_NAMES);blocks=old_blocks+list(BLOCKS)
    old_sources=['frozen_validation.py','compact_regression.py','cnn_controls.py','grain_features.py','latent_features.py','denoised_grain_features.py','candidate_pipeline.py','candidate004_pipeline.py']
    protected={n:sha(_repo_path(ROOT, n)) for n in old_sources}
    plan={'hypothesis':'Connected dark boundary hysteresis and short-gap linking may measure enclosed interiors without proliferating minima-based watershed basins','parameters':PARAMETERS,'development_ids':ids,'outer_folds':{s:int(fid[i]) for i,s in enumerate(ids)},'model':'compact26+connected22 BalancedKernel through unmodified frozen_validation.kernel_fold','inner_grid':{'bandwidth':list(fv.WIDTHS),'alpha':list(fv.ALPHAS),'seed':fv.INNER_SEED,'folds':3},'primary_recipe':{'baseline':.5,'cnn_seed20261003_run009':.25,'combined_kernel':.25},'comparison':'candidate003 same original development folds; original kernel refit is reproducibility control','workers':1,'GPU':False,'new_feature_names':list(FEATURE_NAMES),'blocks':blocks,'source_sha256':{n:sha(_repo_path(ROOT, n)) for n in ['connected_grain_features.py','connected_grain_experiment.py',*old_sources]},'input_sha256':{n:sha(_repo_path(ROOT, n)) for n in ['artifacts/run-001/split_manifest.json','artifacts/run-012/development_features.npz','artifacts/run-012/development_oof.csv','artifacts/run-009/development_oof.csv','artifacts/run-028/qa-selection.json','artifacts/run-028/qa-review.json']},'selection_caveat':'Development400 historically reused. QA images chosen without labels/errors. No independent confirmation, holdout/test access, submission, or fulltrain fit.'}
    write(_repo_path(OUT, 'plan.json'),plan)
    vectors=[];image_hashes={}
    for i,s in enumerate(ids):
        v,n,b=extract_connected(read_image(s,ids));assert n==list(FEATURE_NAMES) and b==list(BLOCKS);vectors.append(v);image_hashes[s]=sha(_repo_path(ROOT, 'data/train', s+'.png'))
        if (i+1)%40==0:print(json.dumps({'extracted':i+1,'seconds':time.perf_counter()-start}),flush=True)
    fresh=np.stack(vectors);X=np.column_stack((old,fresh));assert X.shape==(400,48) and np.isfinite(X).all()
    np.savez_compressed(_repo_path(OUT, 'development_features.npz'),X=X,ids=np.array(ids),groups=groups)
    write(_repo_path(OUT, 'input-provenance.json'),{'image_sha256':image_hashes,'feature_names':names,'blocks':blocks,'feature_cache_sha256':sha(_repo_path(OUT, 'development_features.npz'))})
    feature_seconds=time.perf_counter()-start;new=np.full(400,np.nan);control=np.full(400,np.nan);folds=[]
    for k in range(5):
        tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va])
        control[va],_,control_details=fv.kernel_fold(old,y,groups,old_blocks,tr,va,k)
        np.testing.assert_allclose(control[va],prior_kernel[va],rtol=1e-9,atol=1e-7)
        new[va],_,detail=fv.kernel_fold(X,y,groups,blocks,tr,va,k)
        folds.append({'fold':k,'control':control_details,'combined':detail})
        print(json.dumps({'fold':k,'kernel_rmse':fv.rmse(y[va],new[va]),'combined_blend_rmse':fv.rmse(y[va],.5*baseline[va]+.25*cnn[va]+.25*new[va])}),flush=True)
    values={'candidate003':candidate003,'old_kernel_control':control,'combined_kernel':new,'connected_blend':.5*baseline+.25*cnn+.25*new}
    scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)],'residual_corr_vs003':float(np.corrcoef(y-candidate003,y-p)[0,1])} for n,p in values.items()}
    rng=np.random.default_rng(20261007);delta=[]
    for _ in range(2000):
        ix=rng.integers(0,400,400);delta.append(fv.rmse(y[ix],candidate003[ix])-fv.rmse(y[ix],values['connected_blend'][ix]))
    with (_repo_path(OUT, 'development_oof.csv')).open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness',*values]);w.writerows([s,y[i],*[p[i] for p in values.values()]] for i,s in enumerate(ids))
    assert all(sha(_repo_path(ROOT, n))==h for n,h in protected.items())
    report={**plan,'features':names,'scores':scores,'folds':folds,'feature_seconds':feature_seconds,'elapsed_seconds':time.perf_counter()-start,'conditional_paired_bootstrap_improvement95':np.percentile(delta,[2.5,97.5]).tolist(),'control_max_abs_difference':float(np.max(np.abs(control-prior_kernel))),'new_feature_distribution':{n:np.percentile(fresh[:,j],[0,10,50,90,100]).tolist() for j,n in enumerate(FEATURE_NAMES)},'new_feature_zero_count':{n:int(np.sum(fresh[:,j]==0)) for j,n in enumerate(FEATURE_NAMES)},'feature_cache_sha256':sha(_repo_path(OUT, 'development_features.npz')),'development_oof_sha256':sha(_repo_path(OUT, 'development_oof.csv')),'input_provenance_sha256':sha(_repo_path(OUT, 'input-provenance.json')),'qa_review_sha256':sha(_repo_path(OUT, 'qa-review.json')),'protected_sources_unchanged':True,'promotion':'None; parent must assess and coordinate any repeated folds.'}
    write(_repo_path(OUT, 'report.json'),report);print(json.dumps({'scores':scores,'seconds':report['elapsed_seconds'],'conditional_delta95':report['conditional_paired_bootstrap_improvement95']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['qa','fit']);a=p.parse_args();qa() if a.mode=='qa' else fit()
