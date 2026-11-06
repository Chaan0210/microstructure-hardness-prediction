"""Independent source, mathematical assembly and saved-weight checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import cnn_repair as cr
import sys,json,time,gc
import numpy as np,torch,joblib
from threadpoolctl import threadpool_limits

def sources():
 d=c.inputs();ys=d.pop('y');checks=[]
 for g in d['graphs']:
  k=g['fold']
  for n in g['nodes']:
   p=c.teacher_path(k,n['name']);z=np.load(_repo_io(p));tr=np.array(n['train']);q=np.array(n['query']);assert np.array_equal(tr,z['train']) and np.array_equal(q,z['query']) and set(tr).isdisjoint(q) and (set(tr)|set(q)).isdisjoint(g['V']);ct=json.loads(p.with_name('contract.json').read_text());assert ct['train_ids']==d['ids'][tr].tolist() and ct['query_ids']==d['ids'][q].tolist() and ct['target_sha256']==c.ah(ys[tr]);raw=sum(w*z[a] for a,w in [('tab',.384),('ridge',.128),('logridge',.128),('set',.05),('gmm',.15)])/.84;bad=np.nan_to_num(d['X'][q,list(d['keys']).index('n_obj')],nan=0)<20;raw[bad]=ys[tr].mean();err=float(abs(raw-z['base']).max());assert err<1e-10;checks.append(dict(fold=k,node=n['name'],assembly_maxdiff=err,train_n=len(tr),query_n=len(q)))
 c.save(_repo_io('teacher-source-verification.json'),dict(passed=True,nodes=checks))

def train_repeat():
 d,y=cr.load();z=cr.data(d,y,2,'selector','076');old=c.HERE/'cnn-retarget/seed0/fold2-selector';ck=torch.load(_repo_io(old/'model.pt'),map_location='cpu',weights_only=True);original_inputs=z;outside=np.setdiff1d(np.arange(500),z['train']);altered=y.copy();altered[outside]=999999.;dp={**d,'images':d['images'].copy()};dp['images'][outside]=.123;zz=cr.data(dp,altered,2,'selector','076');assert z['meta']==zz['meta'];assert np.array_equal(z['images'][z['train']],zz['images'][zz['train']]);assert np.array_equal(z['context'],zz['context']);assert all(np.array_equal(z['targets'][a],zz['targets'][a]) for a in z['targets']);cr.ns['ComplementCNN']=cr.Base;start=time.monotonic();path=c.HERE/'verification/poison-training-repeat';path.parent.mkdir(exist_ok=True);cr.fit(zz,'aux',2026101701,path);new=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);diff=max(float(abs(ck['state_dict'][a]-new['state_dict'][a]).max()) for a in ck['state_dict']);assert diff==0,diff;model=cr.Base();model.load_state_dict(new['state_dict']);pr=cr.predict(model,z['images'],z['context'],z['valid'],'aux')*z['meta']['target_std'];ref=np.load(_repo_io(old/'predictions.npz'))['validation_prediction'];pe=float(abs(pr-ref).max());assert pe==0;c.save(_repo_io('training-repeat-verification.json'),dict(passed=True,train_sources=len(z['train']),poisoned_outside_labels_pixels=len(outside),epochs=40,seed=2026101701,max_weight_diff=diff,max_original_query_prediction_diff=pe,seconds=time.monotonic()-start))

def models():
 start=time.monotonic();rows=[]
 for p in sorted((c.HERE/'log-mean').glob('*/*/model.joblib')):
  z=np.load(_repo_io(p.with_name('context.npz')));m=joblib.load(_repo_io(p));pp=m.predict(z['query_X']);err=float(abs(pp-z['prediction_log']).max());assert err<1e-12
  # Preprocessing and coefficient reconstruction from stored, source-restricted context.
  train=np.asarray(z['train']);query=np.asarray(z['query']);assert set(z['source'])<=set(train) and set(train).isdisjoint(query);m2=__import__('sklearn.base',fromlist=['clone']).clone(m).fit(z['augmented_X'],np.log(z['augmented_y']));err2=float(abs(m2.predict(z['query_X'])-pp).max());assert err2<1e-10;rows.append(dict(file=str(p.relative_to(c.HERE)),inference_maxdiff=err,refit_maxdiff=err2))
 assert len(rows)==120,len(rows)
 c.save(_repo_io('log-model-verification.json'),dict(passed=True,models=rows,seconds=time.monotonic()-start))

def assembly():
 records=[]
 for name in ['retarget','log','dense']:
  p=c.HERE/(name+'-oof.npz')
  if not p.exists():continue
  z=np.load(_repo_io(p));r=json.loads((c.HERE/(name+'-results.json')).read_text());y=z['y'];base=z['base082'];hi=z['high'];assert len(set(z['ids']))==400 and set(z['fold'])==set(range(5))
  for key,row in r['results'].items():
   pred=z[key];a=(sum((float(v)-float(t))**2 for v,t in zip(pred,y))/400)**.5;h=(sum((float(v)-float(t))**2 for v,t,yes in zip(pred,y,hi) if yes)/sum(hi))**.5;assert abs(a-row['overall_rmse'])<1e-12 and abs(h-row['high_rmse'])<1e-12
   assert np.array_equal(pred[~hi],base[~hi]);order=sorted(range(400),key=lambda i:(float(base[i])-float(y[i]))**2-(float(pred[i])-float(y[i]))**2,reverse=True)[10:];a0=(sum((float(base[i])-float(y[i]))**2 for i in order)/390)**.5;a1=(sum((float(pred[i])-float(y[i]))**2 for i in order)/390)**.5;assert abs((a0-a1)-row['top10_removed_gain'])<1e-12;records.append(dict(protocol=name,arm=key,RMSE_verified=True,top10_verified=True,unchanged_nongated=True))
 c.save(_repo_io('assembly-verification.json'),dict(passed=True,records=records))

if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):{'sources':sources,'repeat':train_repeat,'models':models,'assembly':assembly}[sys.argv[1]]()
