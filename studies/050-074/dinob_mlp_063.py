"""One fixed same-MLP backbone comparison, with exact ViT-S control replay."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import earlystop_experiment_057 as base
import dinob_mlp_head_063 as h
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/dinob-mlp-063');SEEDS=(20261128,20261205,20261219)

def source_control():
 s=(_repo_path(ROOT, 'auxiliary_head_051.py')).read_text();s=s.replace('def __init__(self):','def __init__(self, input_dim=3456):').replace('nn.Linear(3456,64)','nn.Linear(input_dim,64)').replace('def init_model(seed):','def init_model(seed, input_dim=3456):').replace('return Head()','return Head(input_dim)').replace('m=init_model(seed);initial=','m=init_model(seed, X.shape[1]);initial=');assert s==(_repo_path(ROOT, 'dinob_mlp_head_063.py')).read_text()

def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,v in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==v,n
 source_control();return p

def inputs(split):
 ids,S,y,g,fid,r003,r007,oldmean=base.inputs(split)
 with np.load(_repo_path(ROOT, 'artifacts/dinob-062/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;B=z['X'].astype(float)
 assert B.shape==(400,6912) and np.isfinite(B).all()
 return ids,S,B,y,g,fid,r003,r007,oldmean

def prepare():
 source_control();v=json.loads((_repo_path(ROOT, 'reports/dinob-062-validation.json')).read_text());assert v['passed'] and v['cache_rows_reextracted']==400 and v['cache_max_error']==0
 old=json.loads((base.OUT/'plan.json').read_text());OUT.mkdir(exist_ok=False)
 p={'hypothesis':'Replace frozen ViT-S14 with ViT-B14 under the same candidate007 MLP training recipe. User explicitly requested MLP capacity control after Ridge screen failed. This is a distinct predeclared study, not reversal of Ridge gate.','feature':'Same normalized blocks6/9/12 CLS+patch mean/population std at full504; S3456 versus B6912. First Linear input width necessarily changes, hidden64 and all other settings unchanged. B has more trainable first-layer parameters, so this is recipe control, not parameter-count control.','config':h.CONFIG,'effective_aux_weight':0.,'seeds':list(SEEDS),'head_change':'Exact copy of auxiliary_head_051 with only input width parameterized; validate normalized source equivalence and all15 S endpoint checkpoints against052.','primary':'.75 samefold003 + .25 mean3 B MLP; compare007.','screen':'Original primary gain > conditional bootstrap SD and at least3/5 positive fold gains; then run both existing resplits unchanged. Gate allocates compute, not significance.','confirmation':'Require primary improvement and at least3/5 positive folds in both resplits before promotion consideration. If consistent, inspect one predetermined equal S/B mix (.75 samefold003 + .125 S + .125 B), no weight sweep. Mixed recipe eligible only if all3 partitions improve007 and >=3/5 positive folds each; choose mix only if it beats primary in all3, otherwise primary.','combination_timing':'Equal S/B combination scores computed only after primary passes original screen. No test or holdout input to selection.','scope':'Development400 only until a validated final recipe is locked; full500/test inference permitted only after confirmation. No package installs, payments, uploads.','resource':'Existing uv .venv CPU2threads; benchmark one Bfit and one Sfit; projected30fits <300s, RSS<2GB.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['dinob_mlp_063.py','dinob_mlp_head_063.py','auxiliary_head_051.py','earlystop_experiment_057.py','auxiliary_experiment_051.py']},'inputs':{**old['input_hashes'],**{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/dinob-062/development_features.npz','artifacts/dinob-062/extraction.json','reports/dinob-062-validation.json','reports/seed-052-validation.json']}},'protected':old['protected'],'bias':fv.CAVEAT};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'head_config':h.CONFIG,'seeds':SEEDS}),flush=True)

def prep(X,y):
 sc,ym,ys,edges,bins=h.preprocess(X,y);return sc,ym,ys,torch.tensor(sc.transform(X),dtype=torch.float32),torch.tensor((y-ym)/ys,dtype=torch.float32),torch.tensor(bins),edges

def benchmark():
 check();torch.set_num_threads(2);ids,S,B,y,g,fid,*_=inputs('original');tr=np.flatnonzero(fid!=0);records={};start=time.perf_counter()
 for arm,X in [('controlS',S),('baseB',B)]:
  sc,ym,ys,xt,yt,bt,_=prep(X[tr],y[tr]);t=time.perf_counter();m,r=h.fit(xt,yt,bt,SEEDS[0],0.);r.update(seconds=time.perf_counter()-t,parameters=sum(p.numel() for p in m.parameters()));records[arm]=r
 old=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052/original/fold-0/report.json')).read_text())['models'][str(SEEDS[0])]
 for k in ('initial_state_sha256','batch_order_sha256','final_state_sha256'):assert records['controlS'][k]==old[k]
 assert records['baseB']['batch_order_sha256']==old['batch_order_sha256'];projected=sum(r['seconds'] for r in records.values())*15;report={'models':records,'projected30fit_seconds':projected,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'resource_pass':bool(projected<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2e9),'S_exact_state_reproduction':True,'seconds':time.perf_counter()-start};fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps({k:v for k,v in report.items() if k!='models'}),flush=True)

def comparison(y,ref,new,fid):
 fg=[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20270104).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-ref)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));return {'gain':fv.rmse(y,ref)-fv.rmse(y,new),'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist()}

def run(split):
 check();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,S,B,y,g,fid,r003,r007,oldmean=inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred={arm:np.full((3,400),np.nan) for arm in ('controlS','baseB')};t=time.perf_counter();trainerrors={a:[] for a in pred}
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);fd=out/f'fold-{k}';fd.mkdir();record={'fold':k,'models':{}};oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}/report.json')).read_text())
  for arm,X in [('controlS',S),('baseB',B)]:
   sc,ym,ys,xt,yt,bt,edges=prep(X[tr],y[tr]);tp=[]
   for si,seed in enumerate(SEEDS):
    m,r=h.fit(xt,yt,bt,seed+k,0.);old=oldlog['models'][str(seed)];assert r['batch_order_sha256']==old['batch_order_sha256']
    if arm=='controlS':
     for key in ('initial_state_sha256','final_state_sha256'):assert r[key]==old[key]
    path=fd/f'{arm}-seed-{seed}.pt';torch.save({'model':m.state_dict(),'input_dim':X.shape[1],'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'bin_edges_unused':torch.tensor(edges),'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'seed':seed+k,'config':h.CONFIG,'effective_aux_weight':0.},_repo_io(path));pred[arm][si,va]=h.predict(m,sc.transform(X[va]),ym,ys);tp.append(h.predict(m,xt,ym,ys));r.update(checkpoint_sha256=fv.sha(path),validation_rmse=fv.rmse(y[va],pred[arm][si,va]));record['models'][f'{arm}-{seed}']=r
   trainerrors[arm].extend((np.mean(tp,axis=0)-y[tr]).tolist())
  np.testing.assert_allclose(pred['controlS'][:,va].mean(0),oldmean[va],atol=1e-8,rtol=0);fv.write(fd/'report.json',record);print(json.dumps({'split':split,'fold':k,'seconds':time.perf_counter()-t}),flush=True)
 ps={a:v.mean(0) for a,v in pred.items()};ps.update(reference003=r003,reference007=r007,primary=.75*r003+.25*ps['baseB'],control007=.75*r003+.25*ps['controlS']);np.testing.assert_allclose(ps['control007'],r007,atol=1e-8,rtol=0);comp=comparison(y,r007,ps['primary'],fid);advance=bool(comp['gain']>comp['bootstrap_sd'] and comp['positive_folds']>=3)
 if advance if split=='original' else True:ps['equal_SB']=.75*r003+.125*ps['controlS']+.125*ps['baseB']
 scores={a:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for a,p in ps.items()}
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),y=y,fold=fid,controlS_seeds=pred['controlS'],baseB_seeds=pred['baseB'],**ps)
 report={'split':split,'scores':scores,'comparison':comp,'equal_SB_comparison':comparison(y,r007,ps['equal_SB'],fid) if 'equal_SB' in ps else None,'B_S_residual_correlation':float(np.corrcoef(y-ps['baseB'],y-ps['controlS'])[0,1]),'train_rmse_pooled':{a:float(np.sqrt(np.mean(np.array(v)**2))) for a,v in trainerrors.items()},'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds':time.perf_counter()-t,'control007_max_error':float(abs(ps['control007']-r007).max()),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':advance,'comparison':comp,'primary':'primary'})
 check();print(json.dumps(report),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark}.get(a.mode,lambda:run(a.mode))()
