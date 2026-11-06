"""Strict076 baseline and two fixed, low-cost frozen-CNN inference probes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
PREV=_repo_path(ROOT, 'experiments/078-080-update-20261016')
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'2')
os.environ['MPLCONFIGDIR']=str(_repo_path(HERE, 'mpl-cache'))
sys.path.insert(0,str(PREV))
import study as old
import argparse
import json
import time
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
R4=old.R4;r=old.r
VIEWS=((0,False),(1,False),(0,True),(1,True),(2,False),(3,False),(2,True),(3,True))


def save(name,value):
    p=_repo_path(HERE, name);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def protect():
    path=_repo_path(HERE, 'protected.json')
    if not path.exists():
        records=json.loads((_repo_path(PREV, 'protected.json')).read_text())
        records.update(json.loads((_repo_path(PREV, 'delivery-hashes.json')).read_text()))
        save('protected.json',records)
    for name,digest in json.loads(path.read_text()).items():assert r.digest(_repo_path(ROOT, name))==digest,name


def grouped_linear(d,train,query,labels,variant,log):
    import full_branch as b
    X,y,source,cols=b.linear_training(d,train,labels,variant,True)
    assert set(source)<=set(train) and set(train).isdisjoint(query)
    isreal=np.arange(len(y))<len(train)
    nz=np.r_[d['noise'][train],np.full(len(y)-len(train),99.)]
    grid=[.3,1,2,3,10,30,100] if log else [.1,.3,1,3,10,30,100,1000]
    def fit(a,ix):
        return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
            StandardScaler(),Ridge(alpha=a)).fit(X[ix],np.log(y[ix]) if log else y[ix])
    losses=[]
    for alpha in grid:
        errors=[]
        for tr,va in GroupKFold(5).split(X,y,source):
            assert not set(source[tr])&set(source[va])
            va=va[isreal[va]&(nz[va]>=12)]
            if len(va):
                pred=fit(alpha,tr).predict(X[va]);pred=np.exp(pred) if log else pred
                errors.extend((pred-y[va])**2)
        assert errors;losses.append(float(np.mean(errors)))
    selected=grid[int(np.argmin(losses))]
    model=fit(selected,np.arange(len(y)))
    pred=model.predict(d['variants'][variant][0][query][:,cols])
    return (np.exp(pred) if log else pred),dict(alpha=selected,grid=grid,inner_mse=losses,
        n_real=len(train),n_augmented=len(source)-len(train),source_ids=d['ids'][np.unique(source)].tolist())


def baseline():
    protect();sys.addaudithook(old.no_test)
    import full_branch as b
    d,y=b.load();snap=dict(np.load(_repo_io(R4.parent/'inputs.npz')));dev=d['dev'];fold=d['fold'][0]
    strict=np.load(_repo_io(R4.parent/'scope.npz'));p73=strict['strict_blend'][0]
    gate=d['noise'][dev]>=12;failed=d['fail'][0]
    ridge=np.load(_repo_io(R4.parent/'consistency-strict.npz'))['production'][0].copy()
    logr=np.where(gate,strict['strict'][0],snap['C73_logr'][0])
    rebuilt=.16*snap['C73_cnn'][0]+.384*snap['C73_tab'][0]+.128*ridge+.128*logr+.05*snap['C73_set'][0]+.15*snap['C73_gmm'][0]
    rebuilt[failed]=snap['C73_cnn'][0,failed];assert np.max(np.abs(rebuilt-p73))<1e-10
    sources={};members=[];h=_repo_path(ROOT, 'experiments/push-1300');c=np.load(_repo_io(h/'cumul-train.npz'))
    cp={str(i):j for j,i in enumerate(c['ids'])};cumul=c['X'][[cp[str(i)] for i in d['ids']]][:,[1,4]]
    for slot in ('038','039','044'):
        for short,variant in zip(('u1dm1sz','u1dm2sz','u1dmavg','u1dmavg4'),b.VARIANTS):
            halves={}
            for fs in ('nb','full'):
                tag=f'h9phys-{fs}-044-{short}' if slot=='044' else f'h9-{fs}-{slot}-{short}'
                path=h/f'abl/dev-{tag}.npz';z=np.load(_repo_io(path))
                feature=h/str(z['feature_file']);assert r.digest(feature)==str(z['feature_sha256'])
                F=np.load(_repo_io(feature),allow_pickle=True);fp={str(i):j for j,i in enumerate(F['ids'])}
                xx=F['X'][[fp[str(i)] for i in d['ids']]].astype(float)
                bx,keep=d['banks'][slot,variant];expected=np.c_[bx[:,keep] if fs=='nb' else bx,cumul]
                assert np.array_equal(xx,expected,equal_nan=True)
                pos={str(i):j for j,i in enumerate(z['original_ids'])};ix=[pos[str(i)] for i in d['ids'][dev]]
                assert len(pos)==400 and np.array_equal(z['original_fold'][ix],fold)
                assert np.array_equal(z['original_y'][ix],y[dev])
                halves[fs]=z['original_g'][ix]
                sources[str(path.relative_to(ROOT))]=r.digest(path);sources[str(feature.relative_to(ROOT))]=r.digest(feature)
            members.append(np.where(gate,halves['full'],halves['nb']))
    tab=np.mean(members,axis=0);new_ridge=ridge.copy();new_log=logr.copy();records=[]
    start=time.monotonic()
    for k in range(5):
        tr=dev[fold!=k];ix=np.flatnonzero((fold==k)&gate);query=dev[ix]
        path=_repo_path(HERE, f'linear/fold{k}.npz')
        if path.exists():
            z=np.load(_repo_io(path));assert np.array_equal(z['train'],tr) and np.array_equal(z['query'],query)
            new_ridge[ix]=z['ridge'];new_log[ix]=z['logridge'];records.append(json.loads(path.with_suffix('.json').read_text()))
        else:
            rp,rm=grouped_linear(d,tr,query,y[tr].copy(),b.VARIANTS[2],False)
            logs=[];details=[]
            for v in b.VARIANTS:
                pp,mm=grouped_linear(d,tr,query,y[tr].copy(),v,True);logs.append(np.log(pp));details.append(mm)
            lp=np.exp(np.mean(logs,axis=0));new_ridge[ix]=rp;new_log[ix]=lp
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(_repo_io(path),train=tr,query=query,ridge=rp,logridge=lp)
            record=dict(fold=k,train_ids=d['ids'][tr].tolist(),validation_ids=d['ids'][query].tolist(),
                        ridge=rm,logridge=details,source_sha256=d['source_hashes'])
            save(str(path.relative_to(HERE).with_suffix('.json')),record);records.append(record)
        print('STRICT076 LINEAR',k,'seconds',round(time.monotonic()-start,1),flush=True)
    p76=p73+.384*(tab-snap['C73_tab'][0])+.128*(new_ridge-ridge)+.128*(new_log-logr)
    p76[failed]=p73[failed]
    np.savez_compressed(_repo_path(HERE, 'baseline076.npz'),ids=d['ids'][dev],dev=dev,fold=fold,p73=p73,p76=p76,
        tab073=snap['C73_tab'][0],tab076=tab,ridge073=ridge,ridge076=new_ridge,logridge076=new_log,
        G=snap['C73_cnn'][0],failed=failed,noise=d['noise'][dev])
    save('baseline076.json',dict(strict073=old.metrics(y[dev],p73,fold),strict076=old.metrics(y[dev],p76,fold),
        historical076=old.metrics(y[dev],np.load(_repo_io(h/'dev-combo.npy'))[0],fold),
        source_restricted=True,tab_sources=sources,linear=records,
        limitation='Cached H9 Tab predictions retain their original MPS inference; IDs,folds,labels,features and generation recipe checked. Other components are preserved strict073 predictions.'))
    protect()


def contexts(d,z,version):
    dev=d['dev'];X=d['X'][dev];keys=list(d['keys']);p=z['p73'] if version=='073' else z['p76']
    T=(p-.16*z['G'])/.84;T[z['failed']]=p[z['failed']]
    return np.c_[T,z['tab'+version]-z['ridge'+version],d['noise'][dev],X[:,keys.index('u_ph')],
                 np.log1p(np.maximum(np.nan_to_num(X[:,keys.index('n_obj')],nan=0),0)),X[:,keys.index('coverage')]]


@torch.no_grad()
def predict_views(model,images,contexts):
    model.eval();result=np.empty((len(contexts),8,len(images)),np.float32)
    for first in range(0,len(images),32):
        xb=images[first:first+32]
        for vi,(rot,flip) in enumerate(VIEWS):
            x=np.rot90(xb,rot,axes=(2,3));x=x[:,:,:,::-1] if flip else x
            x=torch.from_numpy(np.ascontiguousarray(x))
            stats=[x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()];h=x
            for block in model.blocks:
                h=block(h);stats.extend([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt()])
            image=torch.cat(stats,1)
            for j,ctx in enumerate(contexts):
                pred=model.main(torch.cat([image,torch.from_numpy(ctx[first:first+32])],1)).squeeze(1)
                result[j,vi,first:first+len(xb)]=pred.numpy()
    return result


def infer():
    protect();sys.addaudithook(old.no_test)
    z=dict(np.load(_repo_path(HERE, 'baseline076.npz')));d=old.e.c.load_data();d.pop('y')
    raw=[contexts(d,z,v) for v in ('073','076')];records=[]
    for k in range(5):
        ix=np.flatnonzero(z['fold']==k);query=z['dev'][ix]
        for seed in (0,1):
            path=R4/f'aux-seed{seed}/fold{k}-final/model.pt'
            ck=torch.load(_repo_io(path),map_location='cpu',weights_only=True);meta=ck['normalization']
            assert set(meta['train_ids']).isdisjoint(d['ids'][query])
            model=old.e.c.ComplementCNN();model.load_state_dict(ck['state_dict'])
            images=((d['images'][query]-meta['image_mean'])/meta['image_std']).astype(np.float32)
            ctx=[]
            for rr in raw:
                rr=np.where(np.isfinite(rr[ix]),rr[ix],np.array(meta['context_median']))
                ctx.append(((rr-np.array(meta['context_mean']))/np.array(meta['context_std'])).astype(np.float32))
            views=predict_views(model,images,ctx)
            four=views[:,:4].mean(1).astype(float)*meta['target_std']
            eight=views.mean(1).astype(float)*meta['target_std']
            ref=np.load(_repo_io(R4/f'aux-seed{seed}/fold{k}-final/predictions.npz'))
            assert np.array_equal(query,ref['validation_index'])
            error=float(np.max(np.abs(four[0]-ref['validation_prediction'])));assert error==0.,error
            out=_repo_path(HERE, f'predictions/fold{k}-seed{seed}.npz');out.parent.mkdir(exist_ok=True)
            np.savez_compressed(_repo_io(out),query=query,dev_index=ix,views=views,four=four,eight=eight,target_std=meta['target_std'])
            records.append(dict(fold=k,seed=seed,old_four_view_replay_max_error=error,model_sha256=r.digest(path)))
            print('FROZEN PROBE',k,seed,'replay',error,flush=True)
    save('inference-verification.json',dict(all_passed=True,records=records,model_updates=False,outer_labels_used=False))
    protect()


def evaluate():
    protect();assert json.loads((_repo_path(HERE, 'inference-verification.json')).read_text())['all_passed']
    d=dict(np.load(_repo_io(R4.parent/'inputs.npz')));z=dict(np.load(_repo_path(HERE, 'baseline076.npz')));y=d['y'][d['dev']]
    weights=json.loads((R4/'noise-calibration/summary.json').read_text())['aux']['selection']
    raw={name:np.zeros((2,400)) for name in ('old4','old8','context4','context8')}
    predictions={name:np.tile(z['p76'],(2,1)) for name in raw};coefs=np.zeros(400)
    for k in range(5):
        for seed in (0,1):
            q=np.load(_repo_path(HERE, f'predictions/fold{k}-seed{seed}.npz'));ix=q['dev_index']
            w=np.array(weights[k]['addition']['weights'])[(z['noise'][ix]>=12).astype(int)]
            w[z['failed'][ix]]=0.;coefs[ix]=w
            for name,key,context in [('old4','four',0),('old8','eight',0),('context4','four',1),('context8','eight',1)]:
                raw[name][seed,ix]=q[key][context]
                predictions[name][seed,ix]+=w*q[key][context]
    mean={name:np.where(z['failed'],z['p76'],z['p76']+coefs*p.mean(0)) for name,p in raw.items()}
    original=np.load(_repo_io(R4/'noise-calibration/aux-oof.npz'))
    assert np.max(np.abs((mean['old4']-z['p76'])-(original['addition']-original['baseline'])))<1e-10
    results={}
    for name,p in mean.items():
        results[name]=r.compare(mean['old4'][None],p[None],y,z['fold'][None])
        results[name]['seed_gains']=[r.rmse(predictions['old4'][j],y)-r.rmse(predictions[name][j],y) for j in (0,1)]
        good=(z['noise']>=12)&~z['failed'];results[name]['high_rmse']=r.rmse(p[good],y[good])
    advance=[n for n,v in results.items() if n!='old4' and v['gain']>=.005 and v['improved_folds']>=3
             and v['conditional_id_bootstrap_gain_ci95'][0]>0]
    draw=np.random.default_rng(2026101627).integers(0,400,(10000,400));a=(mean['old4']-y)**2
    names=['old8','context4','context8'];samples=np.array([np.sqrt(a[draw].mean(1))-np.sqrt(((mean[n]-y)**2)[draw].mean(1)) for n in names])
    gains=np.array([results[n]['gain'] for n in names]);sd=np.maximum(samples.std(1,ddof=1),1e-12)
    crit=float(np.quantile(np.max(np.abs((samples-gains[:,None])/sd[:,None]),axis=0),.95))
    bands={n:[float(g-crit*t),float(g+crit*t)] for n,g,t in zip(names,gains,sd)}
    save('probe-results.json',dict(results=results,advance=advance,simultaneous_bands=bands,
        comparison='Same strict076 base, identical frozen inner coefficients and CNN weights. No outer fitting.',
        limitation='Repeatedly explored400 rows; conditional intervals are not independent generalization evidence.'))
    np.savez_compressed(_repo_path(HERE, 'probe-oof.npz'),ids=z['ids'],fold=z['fold'],y=y,base=z['p76'],**mean)
    print(json.dumps(dict(scores={n:v['candidate']['pooled'] for n,v in results.items()},advance=advance,bands=bands),indent=2),flush=True)
    protect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline','infer','evaluate']);a=p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):{'baseline':baseline,'infer':infer,'evaluate':evaluate}[a.mode]()
