"""Source-nested log retransformation, with a same-data offset control."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import sys,json,time,gc
import numpy as np,torch,joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(c.R4));import full_branch as b
def fit(d,tr,q,labels,v,path):
 X,y,source,cols=b.linear_training(d,tr,labels,v,True);real=np.arange(len(y))<len(tr);noise=np.r_[d['noise'][tr],np.full(len(y)-len(tr),99.)];grid=[.3,1,2,3,10,30,100];loss=[];splits=[]
 assert set(source)<=set(tr) and set(tr).isdisjoint(q)
 def model(a,ix):return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=a)).fit(X[ix],np.log(y[ix]))
 for it,iv in GroupKFold(5).split(X,y,source):
  assert not set(source[it])&set(source[iv]);iv=iv[real[iv]&(noise[iv]>=12)];splits.append((it,iv))
 for a in grid:
  errs=[]
  for it,iv in splits:
   if len(iv):errs.extend((np.exp(model(a,it).predict(X[iv]))-y[iv])**2)
  assert errs;loss.append(float(np.mean(errs)))
 a=grid[int(np.argmin(loss))];m=model(a,np.arange(len(y)));Q=d['variants'][v][0][q][:,cols];p=m.predict(Q);path.mkdir(parents=True);joblib.dump(m,_repo_io(path/'model.joblib'));c.arrays(str((path/'context.npz').relative_to(c.HERE)),train=tr,query=q,train_original_y=labels,augmented_X=X,augmented_y=y,source=source,query_X=Q,prediction_log=p);c.save(_repo_io(str((path/'contract.json').relative_to(c.HERE))),dict(alpha=a,grid=grid,inner_mse=loss,train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist(),source_ids=d['ids'][np.unique(source)].tolist(),train_y_sha256=c.ah(labels),variant=v));del m;return p
def main():
 start=time.monotonic();d,y=b.load();g=c.inputs()['graphs'];base=np.load(_repo_io(c.OLD/'baseline076.npz'));records=[];allpred=np.full(400,np.nan);smear=allpred.copy();offset=allpred.copy();c.save(_repo_io('log-code-lock.json'),dict(code=c.sha(__file__),plan=c.sha(c.HERE/'PLAN.md'),source_hashes=d['source_hashes'],alpha_source=c.sha(c.OLD/'probe.py'),new_factor='mean original-source OOF y / geometric prediction',nested_alpha=True))
 for k,graph in enumerate(g):
  tr=np.array(graph['T']);q=np.array(graph['V']);high=d['noise'][q]>=12;q=q[high];inner=np.full((4,len(tr)),np.nan);outer=[];path=c.HERE/f'log-mean/fold{k}'
  for vi,v in enumerate(b.VARIANTS):
   for j,(it,iv) in enumerate(GroupKFold(5).split(tr,groups=tr)):
    iv=iv[d['noise'][tr[iv]]>=12]
    if len(iv):inner[vi,iv]=fit(d,tr[it],tr[iv],y[tr[it]].copy(),v,path/f'{v}-inner{j}')
   outer.append(fit(d,tr,q,y[tr].copy(),v,path/f'{v}-outer'))
  h=d['noise'][tr]>=12;assert np.isfinite(inner[:,h]).all();ip=np.exp(inner[:,h].mean(0));pp=np.exp(np.mean(outer,0));factor=float(np.mean(y[tr[h]]/ip));bias=float(np.mean(y[tr[h]]-ip));pos={int(x):i for i,x in enumerate(base['dev'])};ix=np.array([pos[int(x)] for x in q]);err=float(np.max(abs(pp-base['logridge076'][ix])));assert err<1e-8,err
  allpred[ix]=pp;smear[ix]=pp*factor;offset[ix]=pp+bias;c.arrays(str((path/'predictions.npz').relative_to(c.HERE)),train=tr,query=q,high_train=tr[h],inner_log=inner,outer_original=pp,smear=pp*factor,offset=pp+bias,factor=np.array(factor),bias=np.array(bias));records.append(dict(fold=k,factor=factor,offset=bias,n_high_train=int(h.sum()),baseline_max_difference=err));print('LOG MEAN',k,'factor',factor,'offset',bias,'s',round(time.monotonic()-start,1),flush=True);gc.collect()
 c.arrays('log-predictions.npz',ids=base['ids'],dev=base['dev'],fold=base['fold'],raw=allpred,smear=smear,offset=offset)
 c.save(_repo_io('log-complete.json'),dict(complete=True,records=records,seconds=time.monotonic()-start,outer_models=20,inner_models=100))
if __name__=='__main__':
 torch.set_num_threads(2);sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):main()
