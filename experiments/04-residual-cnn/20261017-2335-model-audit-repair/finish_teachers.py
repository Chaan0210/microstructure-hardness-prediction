"""Finish only missing T076 contexts; never write in the old experiment."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import sys,json,time,gc,importlib.metadata
import numpy as np,torch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(c.R4));import full_branch as b
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
linear=c.definitions(c.OLD/'probe.py',['grouped_linear'],dict(np=np,GroupKFold=GroupKFold,make_pipeline=make_pipeline,SimpleImputer=SimpleImputer,StandardScaler=StandardScaler,Ridge=Ridge))['grouped_linear']
WEIGHT=c.EXP/'tabpfn-v3/weights/tabpfn-v3.5-20260909.safetensors'
def tab(X,y,Q,seed):
 m=TabPFNRegressor.create_default_for_version(ModelVersion('v3.5'),model_path=str(WEIGHT),device='mps',inference_precision=torch.float32,memory_saving_mode=True,random_state=seed,n_preprocessing_jobs=1,show_progress_bar=False)
 m.fit(X,y.copy());assert m.n_estimators_==8,m.n_estimators_; p=np.asarray(m.predict(Q),float);assert np.isfinite(p).all();del m;gc.collect();torch.mps.empty_cache();return p
def main():
 start=time.monotonic();d,y=b.load();z=np.load(_repo_io(c.EXP/'push-1300/cumul-train.npz'));pos={str(i):j for j,i in enumerate(z['ids'])};extra=z['X'][[pos[str(i)] for i in d['ids']]][:,[1,4]];graphs=c.inputs()['graphs'];sources={};completed=[];fitcount=0
 c.save(_repo_io('teacher-code-lock-v2.json'),dict(source=c.sha(__file__),common=c.sha(c.HERE/'common.py'),plan=c.sha(c.HERE/'PLAN.md'),original_retarg_source=c.sha(c.OLD/'retarget_teachers.py'),weights=c.sha(WEIGHT),environment={x.metadata['Name']:x.version for x in importlib.metadata.distributions()}))
 # Same source size, seed, features and query as one original CPU context.
 f=c.OLD/'teachers076/fold0-deep0';ref=np.load(_repo_io(f/'tab-0-038-u1dm1szH2Z-seed0.npz'));raw,nb=d['banks']['038',b.VARIANTS[0]];X=np.c_[raw,extra][:,np.r_[nb,True,True]];tr,q=ref['train'],ref['query'];p=tab(X[tr],y[tr],X[q],0);err=float(abs(p-ref['prediction']).max());assert err<1e-3,err
 c.save(_repo_io('teacher-device-replay.json'),dict(max_difference_HV=err,tolerance_HV=.001,source=str(f.relative_to(c.ROOT)),train_size=len(tr),query_size=len(q),source_count_matched=True))
 for g in graphs:
  k=g['fold']
  for node in g['nodes']:
   n=node['name'];tr=np.array(node['train']);q=np.array(node['query']);old=c.OLD/f'teachers076/fold{k}-{n}';contract=json.loads((old/'contract.json').read_text()) if (old/'contract.json').exists() else None
   assert not set(tr)&set(q) and not (set(tr)|set(q))&set(g['V'])
   if contract:
    assert contract['train_ids']==d['ids'][tr].tolist() and contract['query_ids']==d['ids'][q].tolist() and contract['target_sha256']==c.ah(y[tr])
   if (old/'predictions.npz').exists():
    zz=np.load(_repo_io(old/'predictions.npz'));assert np.array_equal(zz['train'],tr) and np.array_equal(zz['query'],q)
    sources[str((old/'predictions.npz').relative_to(c.ROOT))]=c.sha(old/'predictions.npz');completed.append([k,n,'reused']);continue
   prior=np.load(_repo_io(c.R4/f'teachers/fold{k}-{n}/predictions.npz'));assert np.array_equal(prior['train'],tr) and np.array_equal(prior['query'],q);out={key:prior[key].copy() for key in b.WEIGHTS};path=c.HERE/f'teachers076/fold{k}-{n}';path.mkdir(parents=True);records=[]
   for noisy in (False,True):
    mask=(d['noise'][q]>=12)==noisy;qi=q[mask]
    if not len(qi):continue
    if noisy:
     out['ridge'][mask],rm=linear(d,tr,qi,y[tr],b.VARIANTS[2],False);logs=[];lmeta=[]
     for v in b.VARIANTS:
      pp,meta=linear(d,tr,qi,y[tr],v,True);logs.append(np.log(pp));lmeta.append(meta)
     out['logridge'][mask]=np.exp(np.mean(logs,0));records.append(dict(ridge=rm,logridge=lmeta))
    members=[]
    for (slot,v),(raw,nb) in d['banks'].items():
     xx=np.c_[raw,extra];cols=np.ones(xx.shape[1],bool) if noisy else np.r_[nb,True,True];xx=xx[:,cols]
     for j in range(3):
      fn=f'tab-{int(noisy)}-{slot}-{v}-seed{j}.npz';src=old/fn
      if src.exists():
       zz=np.load(_repo_io(src));assert np.array_equal(zz['train'],tr) and np.array_equal(zz['query'],qi) and int(zz['seed'])==100*j+k;pp=zz['prediction'];sources[str(src.relative_to(c.ROOT))]=c.sha(src)
      else:
       assert time.monotonic()-start<1800,'30min teacher budget';pp=tab(xx[tr],y[tr],xx[qi],100*j+k);fitcount+=1;c.arrays(str((path/fn).relative_to(c.HERE)),train=tr,query=qi,prediction=pp,seed=100*j+k)
      members.append(pp)
     print('TAB',k,n,int(noisy),slot,v,'new',fitcount,'s',round(time.monotonic()-start,1),flush=True);c.runtime()
    out['tab'][mask]=np.mean(members,0)
   out['base']=sum(b.WEIGHTS[key]*out[key] for key in b.WEIGHTS)/.84;out['base'][d['failed'][q]]=y[tr].mean();out.update(train=tr,query=q,train_ids=d['ids'][tr],query_ids=d['ids'][q]);c.arrays(str((path/'predictions.npz').relative_to(c.HERE)),**out)
   c.save(_repo_io(str((path/'contract.json').relative_to(c.HERE))),dict(train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist(),target_sha256=c.ah(y[tr]),source_sha256=d['source_hashes'],linear=records,changed_device_only='missing members MPS float32, prior members CPU',weights=b.WEIGHTS,fold=k,node=n));completed.append([k,n,'finished'])
 c.save(_repo_io('teachers-complete.json'),dict(complete=True,nodes=completed,new_tab_fits=fitcount,device_check_fits=1,seconds=time.monotonic()-start,reused_sha256=sources));print('TEACHERS COMPLETE',fitcount,flush=True)
if __name__=='__main__':
 assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):main()
