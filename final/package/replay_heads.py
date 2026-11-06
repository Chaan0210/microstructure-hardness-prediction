"""Independent fixed-feature head reproduction for selected #062.

standard: saved G018/G021/Set states; refit frozen linear and GMM recipes.
tab: refit the 72 selected H9s2 TabPFN contexts and infer deployed rows.
source: refit nine source/control TabPFN contexts and infer all rows.
No foundation-model optimization or raw-feature regeneration is claimed here.
"""
from pathlib import Path
import argparse,ast,csv,gc,hashlib,importlib.util,json,os,resource,sys,time,subprocess
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ.setdefault('TABPFN_MODEL_CACHE_SIZE','0')
import numpy as np
from threadpoolctl import threadpool_limits
from reproduce import PROJECT,H,SOURCE,TAGS,dump,sha

def standard(out,data):
    src=PROJECT/'experiments/plateau-break-20261016-2035/reproduce.py'
    source=src.read_text().replace("ROOT/'data/train.csv'","DATA/'train.csv'")
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    ns=dict(Path=Path,HERE=out,ROOT=PROJECT,DATA=data,EXP=PROJECT/'experiments',USED={},np=np,json=json,csv=csv,
            hashlib=hashlib,os=os,sys=sys,gc=gc,importlib=importlib,time=time,resource=resource)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
    def track(path):
        path=Path(path)
        name=str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else 'data/'+str(path.relative_to(data))
        ns['USED'][name]=sha(path)
        return path
    ns['track']=track
    # G018 inference deliberately disables sklearn fitting process-wide.
    # Complete both train-only refits before installing that inference guard.
    ns['linear']();ns['gmm']();ns['neural']()
    dump(out/'scope.json',dict(scope=__doc__,path_adapter="Only label path ROOT/data/train.csv -> --data/train.csv",full_upstream_training=False))

def tabular(out,data,source_mode,max_members,rss_limit_gib,one_new_member=False):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(PROJECT/'experiments/push-300'));import harness
    labels={r['ID']:float(r['hardness']) for r in csv.DictReader((data/'train.csv').open())}
    plans=[];root=PROJECT/'experiments/push-1300'
    components=np.load(PROJECT/'experiments/082-fusion-20261018-0155/reproduction/incumbent-components.npz')
    ids=components['ids'];records=[];values={};start=time.monotonic()
    if source_mode:
        train=np.load(H/'candidate-flow46-v1/training-context.npz');P=np.load(root/'abl/feat-h9s2-full-044-u1dmavg-test.npz')
        assert np.array_equal(P['ids'],ids)
        for tag in ['control']+TAGS:
            X,Q=train['physical'],P['X'].astype(float)
            if tag!='control':
                tr=np.load(SOURCE/f'features/{tag}-train.npz');te=np.load(SOURCE/f'features/{tag}-test.npz')
                assert np.array_equal(tr['ids'],train['ids']) and np.array_equal(te['ids'],ids)
                X=np.c_[X,tr['flow']];Q=np.c_[Q,te['flow']]
            for seed in [0,100,200]:plans.append((tag,seed,X,Q,train['ids'],np.ones(len(ids),bool)))
    else:
        for slot in ['038','039','044']:
            for gate in ['nb','full']:
                active=components['low'] if gate=='nb' else components['high']
                for v in ['u1dm1sz','u1dm2sz','u1dmavg','u1dmavg4']:
                    tr=np.load(root/f'abl/feat-h9s2-{gate}-{slot}-{v}-train.npz');te=np.load(root/f'abl/feat-h9s2-{gate}-{slot}-{v}-test.npz')
                    assert np.array_equal(te['ids'],ids) and np.array_equal(tr['keys'],te['keys'])
                    for seed in [0,100,200]:plans.append((f'{gate}_{slot}_{v}',seed,tr['X'].astype(float),te['X'].astype(float),tr['ids'],active))
    for i,(name,seed,X,Q,tids,active) in enumerate(plans):
        target=np.array([labels[str(k)] for k in tids]);assert len(target)==500
        path=out/f'{name}-seed{seed}.npz'
        reused=path.exists()
        train_hash=hashlib.sha256(X.tobytes()+target.tobytes()).hexdigest()
        query_hash=hashlib.sha256(Q[active].tobytes()).hexdigest()
        if reused:
            z=np.load(path);p=z['prediction'];assert np.array_equal(z['ids'],ids) and int(z['seed'])==seed
            assert np.array_equal(z['active'],active)
            assert str(z['training_values_sha256'])==train_hash
            if 'query_values_sha256' in z:assert str(z['query_values_sha256'])==query_hash
            else:
                # Only the initial checkpoints of this frozen package lack this
                # added query hash. They cannot be reused with another project.
                assert PROJECT==(Path(__file__).resolve().parent/'project')
                if not source_mode:
                    gate,slot,variant=name.split('_',2)
                    fp=root/f'abl/feat-h9s2-{gate}-{slot}-{variant}-test.npz'
                    manifest=json.loads((Path(__file__).resolve().parent/'source-manifest.json').read_text())
                    expected={r['path']:r['sha256'] for r in manifest['files']}
                    assert sha(fp)==expected[str(fp.relative_to(PROJECT))]
            elapsed=float(z['seconds'])
        else:
            tick=time.monotonic();m=harness.tabpfn_new('v3.5',seed,'cpu',n_est=8 if source_mode else None)
            m.set_params(inference_precision=torch.float32,memory_saving_mode=True);m.fit(X,target)
            p=np.full(len(ids),np.nan);p[active]=m.predict(Q[active]);elapsed=time.monotonic()-tick
            assert np.isfinite(p[active]).all()
            np.savez_compressed(path,ids=ids,prediction=p,seed=seed,seconds=elapsed,active=active,
                training_values_sha256=train_hash,query_values_sha256=query_hash)
            del m;gc.collect()
        values.setdefault(name,[]).append(p);records.append(dict(name=name,seed=seed,seconds=elapsed))
        print('HEAD',i+1,'/',len(plans),name,seed,round(elapsed,2),'sec',flush=True)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        records[-1]['peak_rss_bytes']=peak
        records[-1]['checkpoint_reused']=reused
        if not reused:dump(out/f'{name}-seed{seed}-resource.json',dict(peak_rss_bytes=peak,seconds=elapsed,query_sha256=query_hash,train_sha256=train_hash))
        if peak>rss_limit_gib*2**30:
            dump(out/'resource-stop.json',dict(peak_rss_bytes=peak,limit_gib=rss_limit_gib,completed_members=i+1,
                checkpoint_retained=True,model_recipe_changed=False))
            raise RuntimeError('Own head replay exceeded configured RSS limit; completed member checkpoints retained.')
        if one_new_member and not reused and i+1<len(plans):
            dump(out/'worker-progress.json',dict(completed_through=i+1,total=len(plans),peak_rss_bytes=peak))
            return
        if max_members and i+1>=max_members:
            dump(out/'benchmark.json',dict(members=len(records),seconds=time.monotonic()-start,records=records,complete=False));return
    means={k:np.mean(v,0) for k,v in values.items()};output={};errors={}
    if source_mode:
        control=np.load(H/'candidate-flow46-v1/test-predictions.npz');sp=np.load(SOURCE/'test-predictions.npz')
        for name,p in means.items():
            ref=control['control_tab'] if name=='control' else sp[name+'_tab']
            errors[name]=float(np.max(abs(p-ref)));output[name]=p
    else:
        ref=np.load(root/'fin-h9s2-tab-cpu.npz')
        for slot in ['038','039','044']:
            for gate in ['nb','full']:
                key=f'{gate}_{slot}';p=np.mean([means[f'{key}_{v}'] for v in ['u1dm1sz','u1dm2sz','u1dmavg','u1dmavg4']],0)
                errors[key]=float(np.nanmax(abs(p-ref[key])));output[key]=p
    np.savez_compressed(out/'predictions.npz',ids=ids,**output)
    report=dict(complete=True,mode='source' if source_mode else 'tab',members=len(records),records=records,
                max_abs_differences=errors,seconds=time.monotonic()-start,absolute_tolerance_HV=.001,
                tolerance_source='Existing pre-deadline explicit-float32 replay protocol; no change from this result.',
                scope='All fixed TabPFN contexts refit on original500 labels and cached learned features; query inference only.',
                full_upstream_retraining=False,network_used=False)
    dump(out/'verification.json',report);assert max(errors.values())<.001,errors
    print('COMPLETE',json.dumps(errors),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['standard','tab','source'])
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--max-members',type=int,default=0)
    ap.add_argument('--project',type=Path,default=PROJECT)
    ap.add_argument('--rss-limit-gib',type=float,default=4.0,help='Operational memory guard only; does not change model or validation settings.')
    ap.add_argument('--isolated-workers',action='store_true',help='Run one new member per fresh process to bound retained CPU memory.')
    ap.add_argument('--one-new-member',action='store_true',help=argparse.SUPPRESS)
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    PROJECT=a.project.resolve();H=PROJECT/'experiments/supervised-renewal-20261018';SOURCE=H/'candidate-source-flow38-v1'
    if a.mode=='standard' and any(out.iterdir()):raise SystemExit('Use a new empty directory for standard replay; completed or partial results are preserved.')
    if (out/'verification.json').exists():raise SystemExit('Completed output already exists; use a new output directory.')
    for key in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME','HF_HOME']:os.environ[key]=str(out/'tmp')
    (out/'tmp').mkdir(exist_ok=True)
    if a.isolated_workers:
        assert a.mode!='standard' and not a.one_new_member and not a.max_members
        expected=9 if a.mode=='source' else 72
        logs=out/'worker-logs';logs.mkdir(exist_ok=True)
        parent_start=time.monotonic();starting_count=len(list(out.glob('*-seed*.npz')))
        for attempt in range(expected+1):
            count=len(list(out.glob('*-seed*.npz')))
            args=[sys.executable,str(Path(__file__).resolve()),a.mode,'--data',str(Path(a.data).resolve()),'--out',str(out),
                  '--project',str(PROJECT),'--rss-limit-gib',str(a.rss_limit_gib),'--one-new-member']
            with (logs/f'from-{count}.log').open('w') as log:
                subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,TABPFN_MODEL_CACHE_SIZE='0'))
            newer=len(list(out.glob('*-seed*.npz')))
            print('ISOLATED MEMBERS',newer,'/',expected,flush=True)
            if (out/'verification.json').exists():break
            assert newer>count,'No checkpoint progress'
        assert (out/'verification.json').exists()
        report=json.loads((out/'verification.json').read_text())
        report['last_worker_seconds']=report.pop('seconds')
        report.update(isolated_worker_wall_seconds=time.monotonic()-parent_start,starting_cached_members=starting_count,
            sum_member_fit_predict_seconds=sum(r['seconds'] for r in report['records']),operational_rss_limit_GiB=a.rss_limit_gib,
            worker_resource_records=[str(p.name) for p in sorted(out.glob('*-resource.json'))])
        dump(out/'verification.json',report)
    else:
        with threadpool_limits(2):
            standard(out,Path(a.data).resolve()) if a.mode=='standard' else tabular(out,Path(a.data).resolve(),a.mode=='source',a.max_members,a.rss_limit_gib,a.one_new_member)
