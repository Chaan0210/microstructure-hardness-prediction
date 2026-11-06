"""Frozen CNN + matched direct/residual EDF32 heads; internal-only blending."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import ast
import json
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'2')
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import research as r
OLD=r.EXP/'micronet-frozen-205'


def coefficient(base,direction,labels,good):
    p=direction[good];target=(labels-base)[good];den=p@p
    return float(np.clip(p@target/den,0,1)) if den>1e-12 else 0.


def run():
    assert len(list((_repo_path(HERE, 'teachers')).glob('*/details.json')))==30,'Wait for full teacher completion'
    d=dict(np.load(_repo_path(HERE.parent, 'inputs.npz')));dev=d['dev'];y=d['y'];gg=json.loads((_repo_path(HERE, 'label-graph.json')).read_text())
    feature=np.load(_repo_io(OLD/'features.npz'));old=np.load(_repo_io(OLD/'OOF.npz'));manifest=json.loads((OLD/'features.json').read_text())
    assert r.digest(OLD/'features.npz')==manifest['features_SHA256']
    assert np.array_equal(feature['ids'],d['ids'][dev]) and np.array_equal(old['ids'],feature['ids'])
    assert np.array_equal(old['fold'],d['fold'][0]) and np.array_equal(old['y'],y[dev])
    for image_id,digest in manifest['image_sha256'].items():
        assert r.digest(_repo_path(r.ROOT, f'data/train/{image_id}.png'))==digest
    source=(OLD/'ridge.py').read_text();assert source.startswith('from common import np\n')
    source=source.split('\n',1)[1]
    assert all(isinstance(node,ast.FunctionDef) for node in ast.parse(source).body)
    impl={'np':np};exec(compile(source,str(OLD/'ridge.py'),'exec'),impl)
    root=_repo_path(HERE, 'frozen-heads');root.mkdir(exist_ok=True)
    protected=[OLD/'features.npz',OLD/'features.json',OLD/'OOF.npz',OLD/'ridge.py',Path(__file__),_repo_path(HERE, 'AMENDMENT-03.md')]
    hashes={str(p.relative_to(r.ROOT)):r.digest(p) for p in protected};r.save_json(root/'provenance.json',hashes)
    baseline=np.load(_repo_path(HERE.parent, 'scope.npz'))['strict_blend'][0];T=(baseline-.16*d['C73_cnn'][0])/.84
    T[d['fail'][0]]=baseline[d['fail'][0]]
    output={};report={}
    for family in ('imagenet','micronet'):
        X=np.full((500,2048),np.nan);X[dev]=feature[family]
        states={mode:{arm:np.full(400,np.nan) for arm in ('addition','replacement','direction')} for mode in ('direct','residual')}
        weights={mode:[] for mode in states};audits=[]
        for g in gg:
            k=g['fold'];A,B,tr,va=[np.array(g[key]) for key in ('A','B','T','V')]
            fit_results={}
            for stage,kind,train,query in [('selector','deep',A,B),('final','outer',tr,va)]:
                teacher=np.full(500,np.nan)
                for node in g['nodes']:
                    if node['name'].startswith(kind):
                        z=np.load(_repo_path(HERE, f"teachers/fold{k}-{node['name']}/predictions.npz"))
                        assert set(z['train']).isdisjoint(z['query']) and (set(z['train'])|set(z['query']))==set(train)
                        teacher[z['query']]=z['base']
                assert np.isfinite(teacher[train]).all() and set(train).isdisjoint(query)
                geometry=impl['geometry'](X,train)
                # Geometry and alpha use X_train only, not any target or query.
                shifted=X.copy();shifted[query]=10000
                geometric_check=impl['geometry'](shifted,train)
                assert all(np.array_equal(geometry[key],geometric_check[key]) for key in geometry)
                train_design=impl['transform'](X[train],geometry)
                gram=train_design@train_design.T+float(geometry['alpha'])*np.eye(len(train))
                path=root/f'{family}-fold{k}-{stage}';path.mkdir(exist_ok=True)
                np.savez_compressed(_repo_io(path/'geometry.npz'),**geometry)
                for mode in states:
                    targets=np.zeros(500);targets[train]=y[train]-(teacher[train] if mode=='residual' else 0)
                    model=impl['fit'](geometry,targets,train);prediction=impl['predict'](X[query],geometry,model)
                    changed=targets.copy();changed[query]+=10000
                    check=impl['fit'](geometric_check,changed,train)
                    assert all(np.array_equal(model[key],check[key]) for key in model)
                    standardized=(targets[train]-float(model['y_mean']))/float(model['y_std'])
                    dual_coef=train_design.T@solve(gram,standardized,assume_a='pos')
                    dual_prediction=impl['transform'](X[query],geometry)@dual_coef*float(model['y_std'])+float(model['y_mean'])
                    error=float(np.max(np.abs(prediction-dual_prediction)));assert error<1e-8,error
                    np.savez_compressed(_repo_io(path/f'{mode}-model.npz'),**model)
                    restored=dict(np.load(_repo_io(path/f'{mode}-model.npz')))
                    assert np.array_equal(impl['predict'](X[query],geometry,restored),prediction)
                    fit_results[stage,mode]=prediction
                    audits.append(dict(fold=k,stage=stage,mode=mode,df=float(geometry['df']),
                                       alpha=float(geometry['alpha']),dual_max_error=error,poison_exact=True,reload_exact=True,
                                       training_ids=d['ids'][train].tolist(),validation_ids=d['ids'][query].tolist()))
            ib=np.load(_repo_path(HERE, f'teachers/fold{k}-outer0/predictions.npz'));anchor=np.load(_repo_path(HERE, f'anchors/fold{k}.npz'))
            assert np.array_equal(ib['query'],B) and np.array_equal(anchor['B'],B)
            tb=ib['base'];pb=.16*anchor['prediction_B']+.84*tb
            bpos=np.array([list(dev).index(i) for i in B]);vpos=np.array([list(dev).index(i) for i in va]);good=~d['fail'][0,bpos]
            direct_error=float(np.max(np.abs(fit_results['final','direct']-old[family][vpos])))
            assert direct_error<1e-8,('old direct prediction mismatch',family,k,direct_error)
            for mode in states:
                direction_B=fit_results['selector',mode]-(tb if mode=='direct' else 0)
                direction_V=fit_results['final',mode]-(T[vpos] if mode=='direct' else 0)
                a=coefficient(pb,direction_B,y[B],good);b=coefficient(tb,direction_B,y[B],good)
                # Both weights freeze before the outer prediction is evaluated.
                direction_V[d['fail'][0,vpos]]=0.
                states[mode]['direction'][vpos]=direction_V
                states[mode]['addition'][vpos]=baseline[vpos]+a*direction_V
                states[mode]['replacement'][vpos]=T[vpos]+b*direction_V
                weights[mode].append(dict(fold=k,addition=a,replacement=b,inner_validation_ids=d['ids'][B].tolist(),
                                          direct_outer_reproduction_max_error=direct_error))
                np.savez_compressed(_repo_io(path.parent/f'{family}-fold{k}-{mode}-inner.npz'),B=B,base073=pb,non021=tb,
                                    direction=direction_B,good=good,lambda_add=a,lambda_replace=b)
            print('FROZEN HEAD FOLD COMPLETE',family,k,flush=True)
        for mode,predictions in states.items():
            name=f'{family}-{mode}';output[name]=predictions
            metrics={arm:r.compare(baseline[None],predictions[arm][None],y[dev],d['fold'][:1]) for arm in ('addition','replacement')}
            metrics['replacement_vs_deletion']=r.compare(T[None],predictions['replacement'][None],y[dev],d['fold'][:1])
            metrics['inner_weights']=weights[mode];report[name]=metrics
            np.savez_compressed(_repo_io(root/f'{name}-oof.npz'),ids=d['ids'][dev],fold=d['fold'][0],baseline=baseline,non021=T,**predictions)
        for arm in ('addition','replacement'):
            report[f'{family}-residual-vs-direct-{arm}']=r.compare(states['direct'][arm][None],states['residual'][arm][None],y[dev],d['fold'][:1])
        r.save_json(root/f'{family}-verification.json',dict(all_passed=True,fits=audits,raw_images_match_original400=True))
    for name,digest in hashes.items():assert r.digest(_repo_path(r.ROOT, name))==digest
    r.save_json(root/'summary.json',report)
    for name in output:
        print(name,{a:report[name][a]['candidate']['pooled'] for a in ('addition','replacement')},flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
