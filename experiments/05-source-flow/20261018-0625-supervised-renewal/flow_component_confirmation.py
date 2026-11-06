"""Frozen-representation paired component test on all500, two fixed5fold repeats.

This is explicitly NOT a complete-incumbent comparison. It cannot satisfy the
full-system confirmation protocol by itself. H300/P104 historical global
representations remain conditional; all new head/preprocessing fits exclude Q.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME']=str(_repo_path(HERE, 'tmp/huggingface'));os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import json,time,gc,hashlib
import numpy as np
import torch
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));y=d['y'];ids=d['ids'];out=_repo_path(HERE, 'flow-component-confirmation');out.mkdir(exist_ok=True);fp=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(fp));pos={str(v):i for i,v in enumerate(z['ids'])};X=z['X'][[pos[str(v)] for v in ids]].astype(float);z=np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz'));assert np.array_equal(z['ids'],ids);F=z['flow'];fold_seeds=[2026101801,2026101802];init_seeds=[2026101811,2026101823]
 recipe=dict(scope='PairedP104componenttest—notcompleteincumbentconfirmation. All500previouslyexposedlabels;globalrepresentationsconditional.',fold_seeds=fold_seeds,initialization_seeds=init_seeds,n_estimators=8,head_seed_rule='seed+fold;equalmeanofbothseeds',rows_per_training_fold=400,coefficient_fit=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(out/'recipe.json').write_text(json.dumps(recipe,indent=2));sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
 pred={name:np.full((2,500),np.nan) for name in ['control','flow']};folds=np.zeros((2,500),int);records=[]
 for rep,split_seed in enumerate(fold_seeds):
  for fold,(tr,q) in enumerate(KFold(5,shuffle=True,random_state=split_seed).split(X)):
   folds[rep,q]=fold
   for name,A in [('control',X),('flow',np.c_[X,F])]:
    values=[]
    for seed in init_seeds:
     file=out/f'{name}-r{rep}-f{fold}-s{seed}.npz'
     if file.exists():
      z=np.load(_repo_io(file));assert np.array_equal(z['train'],tr) and np.array_equal(z['query'],q);v=z['prediction']
     else:
      start=time.monotonic();m=harness.tabpfn_new('v3.5',seed+fold,'cpu',n_est=8);m.fit(A[tr],y[tr]);v=m.predict(A[q]);np.savez(_repo_io(file),train=tr,query=q,train_ids=ids[tr],query_ids=ids[q],prediction=v);del m;gc.collect();print(name,rep,fold,seed,time.monotonic()-start,flush=True)
     values.append(v)
    pred[name][rep,q]=np.mean(values,0)
   a=np.mean((pred['control'][rep,q]-y[q])**2)**.5;b=np.mean((pred['flow'][rep,q]-y[q])**2)**.5;r=dict(repetition=rep,fold=fold,control_rmse=float(a),flow_rmse=float(b),gain=float(a-b));records.append(r);print('FOLD',json.dumps(r),flush=True);(out/'progress.json').write_text(json.dumps(records,indent=2))
 old=(pred['control']-y)**2;new=(pred['flow']-y)**2;draw=np.random.default_rng(2026101993).integers(500,size=(10000,500));bs=np.sqrt(old[:,draw].mean((0,2)))-np.sqrt(new[:,draw].mean((0,2)));mean_gain=(old-new).mean(0);keep=np.ones(500,bool);keep[np.argsort(mean_gain)[-10:]]=False
 result=dict(control_rmse=float(old.mean()**.5),flow_rmse=float(new.mean()**.5),gain=float(old.mean()**.5-new.mean()**.5),repetition_gains=[float(old[r].mean()**.5-new[r].mean()**.5) for r in range(2)],positive_folds=sum(r['gain']>0 for r in records),paired_original_id_ci95=np.percentile(bs,[2.5,97.5]).tolist(),top10_positive_IDs_removed_gain=float(old[:,keep].mean()**.5-new[:,keep].mean()**.5),folds=records,complete_incumbent_compared=False,full_system_protocol_passed=False,caveat='Thisisacomponenttestwithconditionalhistoricalrepresentations. RepeatedobservationsgroupedbyoriginalID;notindependentnewdataorPrivateprobability.');np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=folds,**pred);(out/'results.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':
 with threadpool_limits(2):main()
