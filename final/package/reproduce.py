"""Frozen #062 reproduction. No downloads, parameter search, or Public-score input.

replay: cached upstream predictions + saved HGB states (exact arithmetic replay).
refit-boosted: retrain the three fixed HGB heads on permitted train features.
cnn: independently infer the two residual CNNs from raw test images and saved context.
These modes are intentionally not described as full upstream retraining.
"""
from pathlib import Path
import argparse, ast, csv, hashlib, json, os, sys, time
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import numpy as np
import joblib
from threadpoolctl import threadpool_limits

PACKAGE=Path(__file__).resolve().parent
PROJECT=PACKAGE/'project'
H=PROJECT/'experiments/supervised-renewal-20261018'
F=H/'final-nonlinear-readouts-20261019-0945'
SOURCE=H/'candidate-source-flow38-v1'
EXPECTED='4e97c57b57eb77b0056552e8649f41bba5a927993b705c4602ffa272c1f3e597'
TAGS=['full500-sourceflow38-s1811','full500-sourceflow38-s1823']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def dump(p,d):
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def csv_read(p):
    rows=list(csv.DictReader(p.open()))
    return np.array([r['ID'] for r in rows]),np.array([float(r['hardness']) for r in rows])

def csv_write(p,ids,y):
    assert len(ids)==len(set(ids))==1000 and np.isfinite(y).all()
    with p.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,[repr(float(v)) for v in y]))
    ri,rp=csv_read(p);assert np.array_equal(ri,ids) and np.array_equal(rp,y)

def base_replay(out):
    """Replay canonical arithmetic and independently check the direct component sum."""
    src=PROJECT/'experiments/plateau-break-20261016-2035/reproduce.py'
    names={'sha','track','save','pack','readcsv','nz','csvout','cached'}
    tree=ast.parse(src.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns=dict(Path=Path,HERE=out,ROOT=PROJECT,EXP=PROJECT/'experiments',USED={},np=np,json=json,csv=csv,hashlib=hashlib)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
    ns['cached']()
    c=dict(np.load(out/'incumbent-components.npz'))
    r4=PROJECT/'experiments/rmse-research-20261015/round4-full073/final500'
    z=np.load(r4/'test-predictions.npz');r=json.loads((r4/'recipe.json').read_text())
    assert np.array_equal(c['ids'],z['ids'])
    p73=csv_read(PROJECT/'artifacts/submission-candidate-073.csv')[1]
    corrected=p73+c['high']*r['production_weights']['addition'][1]*z['seed_corrections'].mean(0)
    corrected[c['failed']]=p73[c['failed']]
    p82=c['p76']+(corrected-p73)
    p83=p82+(csv_read(PROJECT/'artifacts/submission-candidate-079.csv')[1]-csv_read(PROJECT/'artifacts/submission-candidate-078.csv')[1])
    ref83=csv_read(PROJECT/'artifacts/submission-candidate-083.csv')[1]
    assert np.array_equal(p82,csv_read(PROJECT/'artifacts/submission-candidate-082.csv')[1])
    assert np.array_equal(p83,ref83)
    controls=np.load(H/'candidate-flow46-v1/test-predictions.npz')
    assert np.array_equal(p83,controls['base083'])
    forbidden=['submission-candidate-059.csv','submission-candidate-060.csv','make_059.py']
    assert not any(any(bad in k for bad in forbidden) for k in ns['USED'])
    dump(out/'base-ancestry-verification.json',dict(base082_max_difference=0.,base083_max_difference=0.,
        forbidden_legacy_059_060_read=False,used_inputs=ns['USED'],
        scope='Canonical cache arithmetic replay and direct076 component equality; no new raw-feature inference or training.'))
    return c,controls,p83

def training_features(data):
    context=np.load(H/'candidate-flow46-v1/training-context.npz')
    ids=context['ids'];y=context['y'];P=context['physical']
    labels={r['ID']:float(r['hardness']) for r in csv.DictReader((data/'train.csv').open())}
    assert len(ids)==500 and set(ids)==set(labels) and np.array_equal(y,[labels[str(i)] for i in ids])
    X={'control':P}
    for tag in TAGS:
        z=np.load(SOURCE/f'features/{tag}-train.npz');assert np.array_equal(z['ids'],ids)
        X[tag]=np.c_[P,z['flow']]
    return ids,y,X

def final_replay(out,data,refit):
    start=time.monotonic();models={};recipe=json.loads((F/'frozen-recipe.json').read_text())
    if refit:
        from sklearn.ensemble import HistGradientBoostingRegressor
        ids,y,X=training_features(data)
        params={k:v for k,v in recipe['boostedreadout'].items() if k not in ['model','preprocessing','hyperparameters_searched']}
        dest=out/'models';dest.mkdir()
        for name,x in X.items():
            m=HistGradientBoostingRegressor(**params).fit(x,y)
            joblib.dump(dict(kind='boostedreadout',model=m),dest/f'{name}.joblib');models[name]=m
        dump(out/'fit-complete.json',dict(training_rows=500,training_ids=ids.tolist(),test_inputs_read=False,params=params))
    else:
        for name in ['control']+TAGS:
            models[name]=joblib.load(F/f'models/boostedreadout-{name}.joblib')['model']
    # All fitting is complete before the first test feature or prediction is read.
    c,controls,p83=base_replay(out)
    ids=c['ids'];eligible=~c['failed'];assert np.array_equal(eligible,controls['eligible'])
    sp=np.load(SOURCE/'test-predictions.npz');assert np.array_equal(sp['ids'],ids)
    avg={kind:np.mean([sp[tag+'_'+kind] for tag in TAGS],0) for kind in ['tab','raw','log']}
    source=p83+eligible*(.384*(avg['tab']-controls['control_tab'])+.128*(avg['raw']-controls['control_linear'])+.128*(avg['log']-controls['control_log']))
    assert np.array_equal(source,sp['prediction'])
    P=np.load(PROJECT/'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-test.npz')
    assert np.array_equal(P['ids'],ids)
    X={'control':P['X'].astype(float)}
    for tag in TAGS:
        z=np.load(SOURCE/f'features/{tag}-test.npz');assert np.array_equal(z['ids'],ids)
        X[tag]=np.c_[X['control'],z['flow']]
    ps={name:model.predict(X[name]) for name,model in models.items()}
    old=.128*((avg['raw']-controls['control_linear'])+(avg['log']-controls['control_log']))
    final=source+eligible*(-old+.256*(np.mean([ps[tag] for tag in TAGS],0)-ps['control']))
    assert np.array_equal(final[~eligible],p83[~eligible])
    ref=csv_read(F/'candidate-source-flow38-boostedreadout.csv')[1]
    path=out/'candidate-source-flow38-boostedreadout.csv';csv_write(path,ids,final)
    error=float(abs(final-ref).max());digest=sha(path)
    report=dict(mode='refit-boosted' if refit else 'replay',rows=1000,unique_ordered_ids=True,schema=['ID','hardness'],finite=True,
        reference_sha256=EXPECTED,csv_sha256=digest,csv_byte_identical=digest==EXPECTED,max_abs_difference=error,
        rms_difference=float(np.sqrt(np.mean((final-ref)**2))),fallback_unchanged=True,seconds=time.monotonic()-start,
        scope='Three fixed HGB heads newly fit on500 train features; upstream predictions cached.' if refit else 'Saved HGB inference with frozen upstream feature/prediction caches. Not full training.',
        full_upstream_retraining=False,test_fit=False,network_used=False)
    dump(out/'verification.json',report);assert error==0 and digest==EXPECTED,report
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

def cnn(out,data):
    import torch
    from torch import nn
    from PIL import Image
    torch.set_num_threads(2)
    r4=PROJECT/'experiments/rmse-research-20261015/round4-full073'
    src=r4/'cnn_fitter.py';tree=ast.parse(src.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['ComplementCNN','predict']]
    ns=dict(torch=torch,nn=nn,np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
    z=np.load(r4/'final500/test-predictions.npz');ids=z['ids'];active=z['active_indices'];raw=z['context_raw']
    images=np.stack([np.asarray(Image.open(data/f'test/{ids[j]}.png'),np.float32)/255 for j in active])[:,None]
    corr=np.zeros((2,len(ids)))
    for s in [0,1]:
        ck=torch.load(r4/f'final500/models/aux-seed{s}/model.pt',map_location='cpu',weights_only=True);m=ck['normalization']
        net=ns['ComplementCNN']();net.load_state_dict(ck['state_dict'])
        im=((images-m['image_mean'])/m['image_std']).astype(np.float32)
        clean=np.where(np.isfinite(raw),raw,np.array(m['context_median']))
        context=((clean-np.array(m['context_mean']))/np.array(m['context_std'])).astype(np.float32)[active]
        corr[s,active]=ns['predict'](net,im,context,np.arange(len(active)),'aux')*m['target_std']
    err=float(abs(corr-z['seed_corrections']).max());assert err<1e-5
    np.savez_compressed(out/'cnn-predictions.npz',ids=ids,seed_corrections=corr)
    dump(out/'verification.json',dict(raw_test_images=len(active),all_deployed_rows=True,max_abs_difference=err,
        scope='Saved two CNN states + raw test images. Context is frozen upstream prediction/feature cache.',training=False))
    print('CNN raw-image inference max difference',err,flush=True)

def main():
    global PROJECT,H,F,SOURCE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['replay','refit-boosted','cnn']);ap.add_argument('--out',required=True);ap.add_argument('--data')
    ap.add_argument('--project',type=Path,default=PROJECT)
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    PROJECT=a.project.resolve();H=PROJECT/'experiments/supervised-renewal-20261018';F=H/'final-nonlinear-readouts-20261019-0945';SOURCE=H/'candidate-source-flow38-v1'
    data=Path(a.data).resolve() if a.data else None
    if a.mode!='replay' and data is None:ap.error('--data required for this mode')
    for key in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[key]=str(out/'tmp')
    (out/'tmp').mkdir()
    with threadpool_limits(2):
        cnn(out,data) if a.mode=='cnn' else final_replay(out,data,a.mode=='refit-boosted')

if __name__=='__main__':main()
