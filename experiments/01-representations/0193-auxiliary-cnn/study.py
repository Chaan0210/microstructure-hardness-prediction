from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch,time
c.check();assert c.read(R/'preflight.json')['passed'];d=c.data();pred={arm:np.full(400,np.nan) for arm in c.plan['arms']};records=[];start=time.perf_counter()
with c.Monitor('study',1200):
 (R/'original').mkdir(exist_ok=False)
 for k in range(5):
  T,V=c.indices(d,k);pair=[]
  for arm in c.plan['arms']:
   folder=R/'original'/f'fold-{k}-{arm}';p,r=e.fit(d,k,arm,folder);pred[arm][V]=p;pair.append(r);records.append(r)
   if arm=='detach':
    old=torch.load(_repo_io(P/f'artifacts/run-009/fold-{k}.pt'),map_location='cpu',weights_only=True);new=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);model=c.legacy.TinyCNN().to('mps');model.load_state_dict(old['state_dict']);op=e.predict(model,d['images'],V,old['normalization'],'mps');pe=float(abs(op-p).max());se=max(float(abs(v-new['state_dict'][n]).max()) for n,v in old['state_dict'].items());assert pe<1e-3 and se<1e-4,(k,pe,se);c.write(folder/'legacy-equivalence.json',{'passed':True,'prediction_max_error':pe,'state_max_error':se,'original_SHA256':c.sha(P/f'artifacts/run-009/fold-{k}.pt')});del model
   if k==0 and arm=='detach':
    b={'first_fit_seconds':r['seconds'],'projected15fits_seconds':r['seconds']*15,'resource':r['resource'],'first_fit_reused':True};b['passed']=b['projected15fits_seconds']<900;c.write(R/'benchmark.json',b);assert b['passed'];c.emit({'benchmark':b})
  for key in ['initial_state_SHA256','initial_base_SHA256','rng_start','rng_end','input_hashes']:assert pair[0][key]==pair[1][key]==pair[2][key],key
  c.emit({'fold_complete':k,'completed_fits':3*(k+1)})
 assert all(np.isfinite(p).all() for p in pred.values());c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],best021=d['blend_L_MLP'],**pred)
 report={'scores':{a:{'standalone':c.rmse(d['y'],p),'blend10':c.rmse(d['y'],.9*d['blend_L_MLP']+.1*p)} for a,p in pred.items()},'best021':c.rmse(d['y'],d['blend_L_MLP']),'fold_records':records,'OOF_SHA256':c.sha(R/'OOF.npz'),'core_fit_seconds':sum(q['seconds'] for q in records),'study_seconds':time.perf_counter()-start};c.write(R/'report.json',report);c.emit({'main_completed':True,'scores':report['scores'],'fit_seconds':report['core_fit_seconds']})
c.check()
