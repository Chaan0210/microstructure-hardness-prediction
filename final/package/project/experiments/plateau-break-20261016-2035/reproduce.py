"""Frozen incumbent replay. All writes are confined to this experiment."""
from pathlib import Path
import os,sys,json,hashlib,csv,argparse,importlib.util,time,gc,resource
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];EXP=ROOT/'experiments'
sys.dont_write_bytecode=True
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
os.environ['MPLCONFIGDIR']=str(HERE/'mpl-cache');os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1';os.environ['TABPFN_MODEL_CACHE_SIZE']='1'
(HERE/'tmp').mkdir(exist_ok=True)
os.environ['TMPDIR']=str(HERE/'tmp')
import tempfile
tempfile.tempdir=str(HERE/'tmp')
import numpy as np
from threadpoolctl import threadpool_limits
USED={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def track(p):
    p=Path(p);USED[str(p.relative_to(ROOT))]=sha(p);return p
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).absolute();mode=args[1];flags=args[2]
        if ((mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))) and not p.is_relative_to(HERE):raise RuntimeError('Blocked nonexperiment write: '+str(p))
def save(name,obj):
    with (HERE/name).open('x') as f:json.dump(obj,f,indent=2)
def pack(name,**obj):
    p=HERE/name;assert not p.exists();np.savez_compressed(p,**obj)
def nz(p):return dict(np.load(track(p),allow_pickle=True))
def readcsv(tag):
    p=ROOT/f'artifacts/submission-candidate-{tag}.csv';r=list(csv.DictReader(track(p).open()))
    return np.array([v['ID'] for v in r]),np.array([float(v['hardness']) for v in r])
def loadmod(name,p):
    spec=importlib.util.spec_from_file_location(name,track(p));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def csvout(name,ids,p):
    with (HERE/name).open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,[repr(float(x)) for x in p]))

def cached():
    ids,G=readcsv('021');c={t:readcsv(t)[1] for t in ['038','039','044','053','061','073','074','076']}
    original=nz(EXP/'push-1200/fin-none.npz');aug=nz(EXP/'push-1200/fin-d1234n12.npz')
    nb=nz(EXP/'push-1300/fin-nb-partners.npz');B=nz(EXP/'push-1300/fin-srcwB.npz')
    Tn=nz(EXP/'push-1300/fin-nb-tab-cpu.npz');T9=nz(EXP/'push-1300/fin-h9-tab-cpu.npz')
    R=nz(EXP/'probabilistic-grains-900/ridge/final500/predictions.npz');S=nz(EXP/'probabilistic-grains-900/set_model/full500/predictions.npz')
    for z in [original,aug,nb,B,Tn,T9,R,S]:assert np.array_equal(z['ids'],ids)
    failed=R['failed']|S['failed'];high=(np.nan_to_num(aug['noise'],nan=0)>=12)&~failed;low=(~high)&~failed
    full={t:(c[t]-.25*G)/.75 for t in ['038','039','044']}
    p61=c['053']+high*(.128*(aug['ridge']-original['ridge'])+.128*(aug['logridge']-original['logridge']))
    delta=.384*np.mean([Tn['nb_'+t]-full[t] for t in full],axis=0)+.128*(nb['ridge_nb']-nb['ridge_full'])+.128*(nb['logridge_nb']-nb['logridge_full'])+.15*(nb['gmm_nb']-nb['gmm_full'])
    p73=np.where(low,p61+delta,p61)
    p74=p73+high*(.128*(B['ridge']-aug['ridge'])+.128*(B['logridge']-aug['logridge']))
    t0=np.where(low,np.mean([Tn['nb_'+t] for t in full],0),np.mean(list(full.values()),0))
    t9=np.where(low,np.mean([T9['nb_'+t] for t in full],0),np.mean([T9['full_'+t] for t in full],0))
    p76=np.where(failed,p74,p74+.384*(t9-t0))
    checks={t:float(np.max(abs(p-c[t]))) for t,p in [('061',p61),('073',p73),('074',p74),('076',p76)]}
    assert max(checks.values())==0.;csvout('incumbent-076-cache-replay.csv',ids,p76)
    assert sha(HERE/'incumbent-076-cache-replay.csv')==sha(ROOT/'artifacts/submission-candidate-076.csv')
    direct=.16*G+.384*t9+.128*np.where(high,B['ridge'],nb['ridge_nb'])+.128*np.where(high,B['logridge'],nb['logridge_nb'])+.05*S['set_mean_fallback']+.15*np.where(high,nb['gmm_full'],nb['gmm_nb'])
    direct[failed]=G[failed]
    pack('incumbent-components.npz',ids=ids,G=G,Tab=t9,Ridge=np.where(high,B['ridge'],nb['ridge_nb']),logRidge=np.where(high,B['logridge'],nb['logridge_nb']),Set=S['set_mean_fallback'],GMM=np.where(high,nb['gmm_full'],nb['gmm_nb']),failed=failed,high=high,low=low,p76=p76,direct=direct)
    save('cache-replay.json',dict(byte_identical=True,checks=checks,sha256=sha(HERE/'incumbent-076-cache-replay.csv'),
        direct_formula_max_diff=float(np.max(abs(direct-p76))),source_sha256=USED,
        scope='Exact frozen component-output replay; not new raw-image inference or training.'))
    print('CACHE REPLAY EXACT',flush=True)

def linear():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge,RidgeCV
    from sklearn.model_selection import GroupKFold,KFold
    import joblib
    ids,G=readcsv('021');lab={r['ID']:float(r['hardness']) for r in csv.DictReader(track(ROOT/'data/train.csv').open())}
    groups=json.loads(track(EXP/'push-1300/abl/groups.json').read_text());drop=set((groups['het']+' '+groups['intensity']).split())
    variants=['u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z'];pred={};meta=[]
    def pipe(kind,alpha):return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=(kind=='ridge')),StandardScaler(),Ridge(alpha=alpha))
    for condition in ['nb','B']:
        logs=[]
        for v in variants:
            z=nz(EXP/f'push-800/feat-{v}-anghcor-train.npz');q=nz(EXP/f'push-800/feat-{v}-anghcor-test.npz');assert np.array_equal(q['ids'],ids)
            keys=list(z['keys']);trainids=z['ids'];X=z['X'].astype(float);Q=q['X'].astype(float);y=np.array([lab[str(i)] for i in trainids]);source=trainids.copy();nreal=len(y);noise=X[:,keys.index('noise')]*255
            if condition=='B':
                xs=[X];ys=[y];ss=[source]
                for tag in ['deg','deg2','deg3','deg4']:
                    a=nz(EXP/f'push-1200/synthrows-{tag}-{v}-anghcor.npz');use=a['X'][:,keys.index('noise')].astype(float)*255>=12
                    xs.append(a['X'][use].astype(float));ss.append(a['ids'][use]);ys.append(np.array([lab[str(i)] for i in a['ids'][use]]))
                X=np.vstack(xs);y=np.concatenate(ys);source=np.concatenate(ss)
            else:
                cols=np.array([k not in drop for k in keys]);X=X[:,cols];Q=Q[:,cols]
            for kind in (['log','ridge'] if v==variants[2] else ['log']):
                target=np.log(y) if kind=='log' else y
                if condition=='B':
                    grid=[.3,1,2,3,10,30,100] if kind=='log' else [.1,.3,1,3,10,30,100,1000];loss=[]
                    for alpha in grid:
                        errors=[]
                        for tr,va in GroupKFold(5).split(X,y,source):
                            va=va[(va<nreal)];va=va[noise[va]>=12]
                            model=pipe(kind,alpha).fit(X[tr],target[tr],ridge__sample_weight=np.ones(len(tr)))
                            p=model.predict(X[va]);p=np.exp(p) if kind=='log' else p;errors.extend((p-y[va])**2)
                        loss.append(float(np.mean(errors)))
                    alpha=grid[int(np.argmin(loss))];model=pipe(kind,alpha).fit(X,target,ridge__sample_weight=np.ones(len(y)))
                elif kind=='log':alpha=2.;model=pipe(kind,alpha).fit(X,target)
                else:
                    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),RidgeCV(alphas=[.1,1,10,100,1000])).fit(X,target);alpha=float(model[-1].alpha_)
                p=model.predict(Q);modelpath=HERE/f'incumbent-{condition}-{v}-{kind}.joblib';joblib.dump(model,modelpath);assert np.array_equal(joblib.load(modelpath).predict(Q),p)
                if kind=='log':logs.append(np.log(np.exp(p)))
                else:pred[condition+'_ridge']=p
                meta.append(dict(condition=condition,variant=v,kind=kind,alpha=alpha,n_real=nreal,n_rows=len(y),model_sha256=sha(modelpath)))
        pred[condition+'_logridge']=np.exp(np.mean(logs,0))
    nb=nz(EXP/'push-1300/fin-nb-partners.npz');B=nz(EXP/'push-1300/fin-srcwB.npz')
    errors={k:float(np.max(abs(p-(B[k[2:]] if k.startswith('B_') else nb[k[3:]+'_nb'])))) for k,p in pred.items()}
    assert max(errors.values())<1e-8,errors
    pack('incumbent-linear-replay.npz',ids=ids,**pred);save('linear-replay.json',dict(max_differences=errors,models=meta,source_sha256=USED,scope='Reexecuted frozen head-fitting recipe on frozen train-only features; not a fresh full image-model training run.'))
    print('LINEAR REPLAY',errors,flush=True)

def neural():
    import torch
    torch.set_num_threads(1)
    # CPU replay avoids competing for accelerator memory. Historical float32
    # CPU/MPS discrepancy is measured and explicitly bounded at .002 HV here.
    old=EXP/'reproduction-018-176';sys.path.insert(0,str(old));im=loadmod('incumbent_infer018',old/'infer.py')
    f=nz(old/'test-run/test-features.npz');out=im.predict(f,old/'models','cpu');ids=f['ids'];ref18=readcsv('018')[1]
    e18=float(np.max(abs(out['primary']-ref18)));assert e18<.002,e18
    x=nz(EXP/'candidate021-178/test-features.npz');assert np.array_equal(x['ids'],ids)
    manifest=json.loads(track(EXP/'candidate021-178/models/manifest.json').read_text());lp=[]
    for seed in manifest['seed_order']:
        p=EXP/f'candidate021-178/models/L-MLP-{seed}.pt';obj=torch.load(track(p),map_location='cpu',weights_only=True)
        assert sha(p)==manifest['files'][p.name];lp.append(im.head(x['X'],obj))
    G=.9*out['primary']+.1*np.mean(lp,0);e21=float(np.max(abs(G-readcsv('021')[1])));assert e21<.002
    sm=loadmod('incumbent_setfinal',EXP/'probabilistic-grains-900/set_model/finalize500.py')
    bank=sm.load_test_bank(track(EXP/'probabilistic-grains-900/object-bank-test.npz'));assert np.array_equal(bank['ids'],ids)
    values=[]
    for seed in sm.SEEDS:
        path=EXP/f'probabilistic-grains-900/set_model/full500/seed{seed}.pt';ck=torch.load(track(path),map_location='cpu',weights_only=False)
        sm.run.CONFIG=ck['config'];model=sm.run.make_model(bank,'cpu',seed);model.load_state_dict(ck['model_state'])
        dat=sm.InferenceTensorData(bank,ck['context']);values.append(sm.run.predict_y(model,dat,ck['context'],np.arange(1000)))
    ref=nz(EXP/'probabilistic-grains-900/set_model/full500/predictions.npz');raw=np.stack(values)
    es=float(np.max(abs(raw-ref['seed_predictions_raw'])));assert es<.002
    S=np.where(ref['failed'],G,raw.mean(0));pack('incumbent-neural-replay.npz',ids=ids,G=G,Set=S,G018=out['primary'],Lseeds=np.array(lp))
    save('neural-replay.json',dict(G018_cpu_vs_canonical_max_diff=e18,G021_max_diff=e21,grain_set_all_seeds_max_diff=es,
        absolute_tolerance_HV=.002,tolerance_reason='Historical saved float32 CPU/MPS inference tolerance; no fitting; production aggregation reduces this component drift.',
        source_sha256=USED,scope='All1000 saved-weight neural predictions from frozen extracted features. Encoders/features not regenerated. Entire training run not replayed.'))
    print('NEURAL REPLAY',e18,e21,es,flush=True)

def tab():
    import torch
    torch.set_num_threads(1);sys.path.insert(0,str(EXP/'push-300'));import harness
    device=os.environ.get('REPLAY_DEVICE','cpu')
    if device=='mps':
        assert torch.backends.mps.is_available()
        torch.mps.set_per_process_memory_fraction(.3)
    root=EXP/'push-1300';ids,G=readcsv('021');labels={r['ID']:float(r['hardness']) for r in csv.DictReader(track(ROOT/'data/train.csv').open())}
    ckpt=EXP/'tabpfn-v3/weights/tabpfn-v3.5-20260909.safetensors';track(ckpt);track(EXP/'push-300/harness.py')
    saved=nz(root/'fin-h9-tab-cpu.npz');out={};records=[];start=time.monotonic();folder=HERE/'tab-replay-float32';folder.mkdir(exist_ok=True)
    components=nz(HERE/'incumbent-components.npz')
    for slot in ['038','039','044']:
        for gate in ['nb','full']:
            active=components['low'] if gate=='nb' else components['high']
            members=[]
            for variant in ['u1dm1sz','u1dm2sz','u1dmavg','u1dmavg4']:
                t=nz(root/f'abl/feat-h9-{gate}-{slot}-{variant}-train.npz');q=nz(root/f'abl/feat-h9-{gate}-{slot}-{variant}-test.npz')
                assert np.array_equal(q['ids'],ids) and np.array_equal(t['keys'],q['keys']);y=np.array([labels[str(i)] for i in t['ids']]);seeds=[]
                for seed in [0,100,200]:
                    path=folder/f'{slot}-{gate}-{variant}-{seed}.npz';contract=dict(train_sha=sha(root/f'abl/feat-h9-{gate}-{slot}-{variant}-train.npz'),test_sha=sha(root/f'abl/feat-h9-{gate}-{slot}-{variant}-test.npz'),seed=seed,memory_saving_mode=True,inference_precision='float32')
                    legacy=HERE/'tab-replay'/path.name
                    if not path.exists() and legacy.exists():
                        z=np.load(legacy)
                        if 'device' not in z or str(z['device'])=='cpu':
                            old=json.loads(str(z['contract']));assert all(old[k]==contract[k] for k in old)
                            np.savez_compressed(path,ids=ids,pred=z['pred'],seconds=z['seconds'],contract=json.dumps(contract),device='cpu',legacy_sha256=sha(legacy))
                    if path.exists():
                        z=np.load(path);assert json.loads(str(z['contract']))==contract;p=z['pred'];elapsed=float(z['seconds'])
                    else:
                        tick=time.monotonic();model=harness.tabpfn_new('v3.5',seed,device);model.set_params(memory_saving_mode=True,inference_precision=torch.float32);model.fit(t['X'].astype(float),y)
                        p=np.full(len(ids),np.nan);p[active]=model.predict(q['X'].astype(float)[active]);elapsed=time.monotonic()-tick
                        assert np.isfinite(p[active]).all();np.savez_compressed(path,ids=ids,pred=p,seconds=elapsed,contract=json.dumps(contract),device=device,active=active);del model;gc.collect()
                        if device=='mps':torch.mps.empty_cache()
                    seeds.append(p);records.append(dict(slot=slot,gate=gate,variant=variant,seed=seed,seconds=elapsed))
                    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    print('TAB REPLAY',len(records),'/72','seconds',round(time.monotonic()-start,1),'peak_MiB',round(peak/2**20),flush=True)
                    if peak>4*2**30:raise RuntimeError('Own replay memory limit reached; checkpoints retained, other jobs untouched')
                members.append(np.mean(seeds,0))
            key=gate+'_'+slot;out[key]=np.mean(members,0);err=float(np.max(abs(out[key][active]-saved[key][active])));assert err<.001,(key,err)
    pack('incumbent-tab-replay.npz',ids=ids,**out);save('tab-replay.json',dict(records=records,source_sha256=USED,max_differences={k:float(np.nanmax(abs(p-saved[k]))) for k,p in out.items()},
        seconds=time.monotonic()-start,device=device,scope='All72 frozen TabPFN members replayed on their deployed query rows from original500 context and fixed pretrained weights/features. Inactive rows are NaN. Initial completed CPU calls retained. No foundation-weight training or test fitting.',
        component_tolerance_HV=.001,tolerance_reason='Explicit float32 on both backends. One-model probe bounds backend drift at .000146 HV versus .100 under automatic reduced precision. Bank tolerance .001 HV bounds final-ensemble RMSE perturbation from Tab at .000384 HV, well below .03 advancement criterion; actual maximum also reported.'))

def gmm():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    import joblib
    source=EXP/'push-900/gmm_partner.py';ns={'__file__':str(source),'__name__':'incumbent_gmm'}
    exec(compile(track(source).read_text().split('mode = sys.argv[1]')[0],str(source),'exec'),ns)
    tr=nz(EXP/'push-800/feat-u1dmavgH2Z-anghcor-train.npz');te=nz(EXP/'push-800/feat-u1dmavgH2Z-anghcor-test.npz');labs={r['ID']:float(r['hardness']) for r in csv.DictReader(track(ROOT/'data/train.csv').open())};y=np.array([labs[str(i)] for i in tr['ids']])
    gt=ns['gmm_feats'](np.load(track(EXP/'probabilistic-grains-900/object-bank-train.npz'),allow_pickle=True),tr['ids'].tolist())
    ge=ns['gmm_feats'](np.load(track(EXP/'probabilistic-grains-900/object-bank-test.npz'),allow_pickle=True),te['ids'].tolist())
    groups=json.loads(track(EXP/'push-1300/abl/groups.json').read_text());drop=set((groups['het']+' '+groups['intensity']).split());out={};meta=[]
    for gate in ['nb','full']:
        cols=np.array([gate=='full' or k not in drop for k in tr['keys']]);X=np.c_[tr['X'].astype(float)[:,cols],gt];Q=np.c_[te['X'].astype(float)[:,cols],ge];loss=[]
        for a in ns['ALPHAS']:
            err=[]
            for i,j in KFold(3,shuffle=True,random_state=19001).split(X):err.extend((ns['ridge'](a).fit(X[i],y[i]).predict(X[j])-y[j])**2)
            loss.append(float(np.mean(err)))
        alpha=ns['ALPHAS'][int(np.argmin(loss))];model=ns['ridge'](alpha).fit(X,y);out['gmm_'+gate]=model.predict(Q);joblib.dump(model,HERE/f'incumbent-gmm-{gate}.joblib');meta.append(dict(gate=gate,alpha=alpha,inner_mse=loss))
    ref=nz(EXP/'push-1300/fin-nb-partners.npz');errors={k:float(np.max(abs(p-ref[k]))) for k,p in out.items()};assert max(errors.values())<1e-8,errors
    pack('incumbent-gmm-replay.npz',ids=te['ids'],**out);save('gmm-replay.json',dict(max_differences=errors,models=meta,source_sha256=USED,scope='Frozen per-image object measurement and train-only head fitting recipe replayed.'))
    print('GMM REPLAY',errors,flush=True)

def final():
    c=nz(HERE/'incumbent-components.npz');l=nz(HERE/'incumbent-linear-replay.npz');n=nz(HERE/'incumbent-neural-replay.npz');t=nz(HERE/'incumbent-tab-replay.npz');g=nz(HERE/'incumbent-gmm-replay.npz')
    for z in [l,n,t,g]:assert np.array_equal(z['ids'],c['ids'])
    tab=np.where(c['high'],np.mean([t['full_'+s] for s in ['038','039','044']],0),np.mean([t['nb_'+s] for s in ['038','039','044']],0))
    p=.16*n['G']+.384*tab+.128*np.where(c['high'],l['B_ridge'],l['nb_ridge'])+.128*np.where(c['high'],l['B_logridge'],l['nb_logridge'])+.05*n['Set']+.15*np.where(c['high'],g['gmm_full'],g['gmm_nb']);p[c['failed']]=n['G'][c['failed']]
    error=float(np.max(abs(p-c['p76'])));assert error<.002,error;csvout('incumbent-076-model-replay.csv',c['ids'],p)
    save('model-replay.json',dict(max_absolute_difference=error,rms_difference=float(np.sqrt(np.mean((p-c['p76'])**2))),
        absolute_tolerance_HV=.002,tolerance_reason='Conservative inherited CPU/MPS float32 component-inference bound; measured actual discrepancy reported separately. Cached historical assembly remains byte exact.',
        all1000=True,all_final_head_components_replayed=True,feature_caches_reused=True,raw_image_feature_regeneration=False,full_training_run_reproduced=False,
        source_sha256=USED,sha256=sha(HERE/'incumbent-076-model-replay.csv')))
    print('FULL HEAD REPLAY',error,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['cached','linear','gmm','neural','tab','final']);args=a.parse_args();sys.addaudithook(guard)
    with threadpool_limits(limits=1):globals()[args.action]()
