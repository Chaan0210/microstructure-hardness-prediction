"""Two precommitted nonlinear readouts. Fit on train only, then infer/replay."""
from pathlib import Path
import os,sys
OUT=Path(__file__).resolve().parent;HERE=OUT.parent;ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
for key in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[key]=str(HERE/'tmp')
import csv,json,hashlib,datetime,importlib.metadata
import numpy as np
import joblib
from scipy.linalg import solve
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def kernel(A,B):
    d=A.shape[1];dot=A@B.T
    sq=np.maximum(np.sum(A*A,1)[:,None]+np.sum(B*B,1)[None]-2*dot,0.)
    return dot/d+np.exp(-sq/(2*d))
def feature(source,tag,split,ids,recipe_hash):
    p=source/f'features/{tag}-{split}.npz';meta=json.loads(p.with_name(f'{tag}-{split}-complete.json').read_text())
    assert meta['complete'] and meta['recipe_sha256']==recipe_hash and meta['feature_sha256']==sha(p)
    z=np.load(p);assert np.array_equal(z['ids'],ids) and z['flow'].shape==(len(ids),38)
    return z['flow']
def fit(kind,X,y,spec):
    assert not np.isinf(X).any()
    if kind=='hybridkernel':
        transform=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler())
        Z=transform.fit_transform(X);K=kernel(Z,Z);alpha=100/Z.shape[1];K.flat[::len(K)+1]+=alpha
        return dict(kind=kind,transform=transform,Z=Z,coef=solve(K,y-y.mean(),assume_a='pos'),mean=float(y.mean()),alpha=alpha)
    params={k:v for k,v in spec.items() if k not in ['model','preprocessing','hyperparameters_searched']}
    model=HistGradientBoostingRegressor(**params).fit(X,y)
    return dict(kind=kind,model=model)
def predict(model,X):
    if model['kind']=='hybridkernel':return kernel(model['transform'].transform(X),model['Z'])@model['coef']+model['mean']
    return model['model'].predict(X)
def main():
    r=json.loads((OUT/'frozen-recipe.json').read_text());source=HERE/r['source_directory'];recipe_hash=r['source_recipe_sha256']
    assert sha(source/'frozen-recipe.json')==recipe_hash
    tc=HERE/r['training_context'];assert sha(tc)==r['training_context_sha256'];z=np.load(tc)
    ids,y,P=z['ids'],z['y'],z['physical'];assert len(ids)==len(set(ids))==500
    labels={v['ID']:float(v['hardness']) for v in csv.DictReader((ROOT/'data/train.csv').open())}
    assert np.array_equal(y,[labels[str(i)] for i in ids])
    train={'control':P}
    for tag in r['encoder_tags']:train[tag]=np.c_[P,feature(source,tag,'train',ids,recipe_hash)]
    dest=OUT/'models';dest.mkdir(exist_ok=False);fit_hashes={}
    for kind in r['hypotheses_fixed_before_either_public_score']:
        for name,X in train.items():
            model=fit(kind,X,y,r[kind]);path=dest/f'{kind}-{name}.joblib';joblib.dump(model,path);fit_hashes[path.name]=sha(path)
            print('FITTED',kind,name,flush=True)
    save(OUT/'fit-complete.json',dict(complete=True,training_rows=500,training_ids=ids.tolist(),model_hashes=fit_hashes,
        test_inputs_read=False,recipe_sha256=sha(OUT/'frozen-recipe.json'),code_sha256=sha(Path(__file__)),
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn']}))
    # No test values enter fitting. Both readout recipes and allsixstates are fixed above.
    sp=np.load(source/'test-predictions.npz');test_ids=np.array([v['ID'] for v in csv.DictReader((ROOT/'data/sample_submission.csv').open())])
    assert len(test_ids)==len(set(test_ids))==1000 and np.array_equal(sp['ids'],test_ids)
    original=source/'candidate-source-flow38.csv';assert sha(original)==r['source_csv_sha256']
    controls_path=HERE/r['controls'];assert sha(controls_path)==r['controls_sha256'];controls=np.load(controls_path)
    assert np.array_equal(controls['ids'],test_ids)
    base_recipe=json.loads((source/'frozen-recipe.json').read_text());physical=np.load(ROOT/base_recipe['physical_test'])
    assert np.array_equal(physical['ids'],test_ids) and physical['keys'].tolist()==base_recipe['physical_keys']
    test={'control':physical['X'].astype(float)}
    for tag in r['encoder_tags']:test[tag]=np.c_[test['control'],feature(source,tag,'test',test_ids,recipe_hash)]
    eligible=sp['eligible'];assert np.array_equal(eligible,controls['eligible'])
    old_delta=.128*((np.mean([sp[tag+'_raw'] for tag in r['encoder_tags']],0)-controls['control_linear'])+
                    (np.mean([sp[tag+'_log'] for tag in r['encoder_tags']],0)-controls['control_log']))
    for kind in r['hypotheses_fixed_before_either_public_score']:
        runs=[];input_predictions={}
        for repetition in range(2):
            ps={name:predict(joblib.load(dest/f'{kind}-{name}.joblib'),X) for name,X in test.items()}
            p=sp['prediction']+eligible*(-old_delta+.256*(np.mean([ps[tag] for tag in r['encoder_tags']],0)-ps['control']))
            assert np.isfinite(p).all() and np.array_equal(p[~eligible],sp['prediction'][~eligible])
            path=OUT/f'candidate-source-flow38-{kind}{"-replay" if repetition else ""}.csv';assert not path.exists()
            with path.open('x',newline='') as f:
                w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(test_ids,[repr(float(v)) for v in p]))
            rows=list(csv.DictReader(path.open()));assert len(rows)==1000 and [v['ID'] for v in rows]==test_ids.tolist()
            assert np.array_equal(np.array([float(v['hardness']) for v in rows]),p)
            runs.append((p,sha(path)));input_predictions=ps
        assert np.array_equal(runs[0][0],runs[1][0]) and runs[0][1]==runs[1][1]
        np.savez(OUT/f'{kind}-predictions.npz',ids=test_ids,prediction=runs[0][0],**input_predictions)
        save(OUT/f'{kind}-verification.json',dict(complete=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             rows=1000,ordered_unique_ids=True,schema=['ID','hardness'],finite=True,fallback_unchanged=True,
             saved_model_replay_max_difference=0.,csv_byte_identical=True,csv_sha256=runs[0][1],
             recipe_sha256=sha(OUT/'frozen-recipe.json'),no_test_fit_or_selection=True,public_result=None))
        print('VERIFIED',kind,runs[0][1],flush=True)
    assert sha(original)==r['source_csv_sha256'];print('BOTH_CANDIDATES_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(2):main()
