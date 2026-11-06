"""Fit the frozen flow46-v1 candidate on all500, then infer/replay its CSV.

Test inputs are never used by the fit action. Predict/replay require the exact
frozen recipe and fitted-state artifacts. No submission or selection mutation.
"""
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'candidate-flow46-v1'
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[k]=str(HERE/'tmp')
os.environ['HF_HOME']=str(HERE/'tmp/huggingface');os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import argparse,csv,hashlib,json,time,gc,importlib.metadata
import numpy as np
import torch,joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
def recipe():
 r=json.loads((OUT/'frozen-recipe.json').read_text());assert r['status'].startswith('frozen_')
 for name,digest in r['source_files'].items():assert sha(HERE/name)==digest
 assert sha(ROOT/r['baseline_csv'])==r['baseline_sha256'];return r

def fit():
 r=recipe();d=np.load(HERE/'data.npz');ids=d['ids'];y=d['y'];z=np.load(ROOT/r['physical_train']);pos={str(v):i for i,v in enumerate(z['ids'])};X=z['X'][[pos[str(v)] for v in ids]].astype(float);assert z['keys'].tolist()==r['physical_keys']
 h=np.load(ROOT/r['flow_training_cache']);assert np.array_equal(h['ids'],ids) and h['flow_keys'].tolist()==r['flow_keys'];F=h['flow'];A={'control':X,'flow':np.c_[X,F]};assert len(ids)==len(y)==500 and len(set(ids))==500
 models=OUT/'models';models.mkdir(exist_ok=True);np.savez(OUT/'training-context.npz',ids=ids,y=y,physical=X,flow=F)
 sys.path.insert(0,str(ROOT/'experiments/push-300'));import harness
 from tabpfn import TabPFNRegressor
 checks=[];start=time.monotonic()
 for name,T in A.items():
  for kind in ['linear','log']:
   path=models/f'{name}-{kind}.joblib'
   if not path.exists():
    m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.));m.fit(T,y if kind=='linear' else np.log(y));joblib.dump(m,path)
  for seed in [0,100,200]:
   path=models/f'{name}-seed{seed}.tabpfn_fit'
   if path.exists():continue
   m=harness.tabpfn_new('v3.5',seed,'cpu',n_est=8);m.fit(T,y);m.save_fit_state(path);probe=m.predict(T[:3]);del m;gc.collect();restored=TabPFNRegressor.load_from_fit_state(path,device='cpu');replay=restored.predict(T[:3]);error=float(abs(probe-replay).max());assert error<1e-5;checks.append(dict(model=path.name,roundtrip_max_difference=error,technical_probe_only_not_validation=True));del restored;gc.collect();print('FITTED',name,seed,flush=True)
 version={k:importlib.metadata.version(k) for k in ['torch','tabpfn','numpy','scikit-learn','scipy']};manifest={str(p.relative_to(OUT)):sha(p) for p in models.iterdir() if p.is_file()}
 save(OUT/'fit-complete.json',dict(complete=True,original_training_rows=500,all_training_ids=ids.tolist(),test_inputs_read=False,seconds=time.monotonic()-start,versions=version,roundtrip_checks=checks,model_hashes=manifest,training_context_sha256=sha(OUT/'training-context.npz'),recipe_sha256=sha(OUT/'frozen-recipe.json'),implementation_sha256=sha(Path(__file__))));print('FIT_COMPLETE',flush=True)

def predict(replay=False):
 r=recipe();meta=json.loads((OUT/'fit-complete.json').read_text());assert meta['recipe_sha256']==sha(OUT/'frozen-recipe.json')
 for name,digest in meta['model_hashes'].items():assert sha(OUT/name)==digest
 sample=list(csv.DictReader((ROOT/'data/sample_submission.csv').open()));ids=np.array([x['ID'] for x in sample]);assert len(ids)==1000 and len(set(ids))==1000
 z=np.load(ROOT/r['physical_test']);assert z['keys'].tolist()==r['physical_keys'] and np.array_equal(z['ids'],ids);X=z['X'].astype(float);h=np.load(OUT/'test-flow-features.npz');assert np.array_equal(h['ids'],ids) and h['flow_keys'].tolist()==r['flow_keys'];F=h['flow'];assert F.shape==(1000,46) and np.isfinite(F).all()
 rows=list(csv.DictReader((ROOT/r['baseline_csv']).open()));assert np.array_equal([x['ID'] for x in rows],ids);base=np.array([float(x['hardness']) for x in rows]);assert sha(ROOT/r['failure_cache'])==r['failure_cache_sha256'];z=np.load(ROOT/r['failure_cache']);assert np.array_equal(z['ids'],ids);eligible=~z['failed'].astype(bool)
 sys.path.insert(0,str(ROOT/'experiments/push-300'));import harness
 harness.tabpfn_new('v3.5',0,'cpu',n_est=8) # Install the existing offline network guard.
 from tabpfn import TabPFNRegressor
 ps={};start=time.monotonic()
 for name,T in [('control',X),('flow',np.c_[X,F])]:
  vals=[]
  for seed in [0,100,200]:
   model=TabPFNRegressor.load_from_fit_state(OUT/f'models/{name}-seed{seed}.tabpfn_fit',device='cpu');vals.append(model.predict(T));del model;gc.collect();print('INFERRED',name,seed,flush=True)
  ps[name+'_tab']=np.mean(vals,0)
  for kind in ['linear','log']:
   m=joblib.load(OUT/f'models/{name}-{kind}.joblib');v=m.predict(T);ps[name+'_'+kind]=v if kind=='linear' else np.exp(v)
 p=base+eligible*(.384*(ps['flow_tab']-ps['control_tab'])+.128*(ps['flow_linear']-ps['control_linear'])+.128*(ps['flow_log']-ps['control_log']))
 assert p.shape==(1000,) and np.isfinite(p).all();assert np.array_equal(p[~eligible],base[~eligible]);target=OUT/('candidate-flow46-v1-replay.csv' if replay else 'candidate-flow46-v1.csv')
 with target.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,[repr(float(v)) for v in p]))
 report=dict(rows=1000,ids_unique=True,ids_match_sample_order=True,schema=['ID','hardness'],finite_predictions=True,frozen_failure_fallback_unchanged=True,source_recipe_sha256=sha(OUT/'frozen-recipe.json'),csv_sha256=sha(target),seconds=time.monotonic()-start,no_upload=True,no_selection_change=True)
 if replay:
  original=np.load(OUT/'test-predictions.npz');err=float(abs(p-original['prediction']).max());assert err<1e-5;report.update(replay_max_difference=err,csv_byte_identical=sha(target)==sha(OUT/'candidate-flow46-v1.csv'));save(OUT/'replay-verification.json',report)
 else:np.savez(OUT/'test-predictions.npz',ids=ids,prediction=p,base083=base,eligible=eligible,**ps);save(OUT/'csv-verification.json',report)
 print('CSV_VERIFIED',json.dumps(report),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','predict','replay']);a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
 with threadpool_limits(2):fit() if a.action=='fit' else predict(a.action=='replay')
