from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mixup_experiment_061 as e
import numpy as np,torch
fv=e.fv;OUT=e.OUT

def main():
 start=time.perf_counter();plan=e.check();torch.set_num_threads(2);results={};count=0;replay_error=None
 for split in ('original','20261031','20261101'):
  out=OUT/split
  if not out.exists():continue
  ids,X,y,g,fid,r003,r007,oldmean=e.base.inputs(split);report=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==report['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz')) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['fold'],fid);sp=z['seed_predictions'].copy();new=z['primary'].copy()
  errors=[];train=[];bins=np.empty(400,dtype=int);outside=0
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);sc,ym,ys,xt,yt=e.base.prep(X[tr],y[tr]);log=json.loads((out/f'fold-{k}'/'report.json').read_text());oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}', 'report.json')).read_text());tp=[];bins[va]=np.searchsorted(np.quantile(y[tr],[.2,.8]),y[va])
   for si,seed in enumerate(e.head.SEEDS):
    p=out/f'fold-{k}'/f'seed-{seed}.pt';rec=log['models'][str(seed)];old=oldlog['models'][str(seed)];assert fv.sha(p)==rec['checkpoint_sha256'];a=torch.load(_repo_io(p),weights_only=True,map_location='cpu');assert a['train_ids']==[ids[i] for i in tr] and a['validation_ids']==[ids[i] for i in va];assert a['seed']==seed+k and a['target_mean']==ym and a['target_std']==ys;np.testing.assert_array_equal(a['scaler_mean'].numpy(),sc.mean_);np.testing.assert_array_equal(a['scaler_scale'].numpy(),sc.scale_);np.testing.assert_array_equal(a['scaler_var'].numpy(),sc.var_);assert rec['initial_hash']==old['initial_state_sha256'] and rec['order_hash']==old['batch_order_sha256'];assert e.base.h.state_hash(a['state'])==rec['final_hash'];pv=e.head.predict(a['state'],sc.transform(X[va]),ym,ys);np.testing.assert_array_equal(pv,sp[si,va]);errors.append(float(abs(pv-sp[si,va]).max()));tp.append(e.head.predict(a['state'],xt,ym,ys));count+=1
    if split=='original' and k==0 and si==0:
     states,rr=e.head.fit(xt,yt,seed);assert rr['final_hash']==rec['final_hash'] and rr['mixup_hash']==rec['mixup_hash'];rp=e.head.predict(states[100],sc.transform(X[va]),ym,ys);np.testing.assert_array_equal(rp,pv);replay_error=float(abs(rp-pv).max())
   avg=np.mean(tp,axis=0);train.extend((avg-y[tr]).tolist());p=sp[:,va].mean(0);outside+=int(((p<y[tr].min())|(p>y[tr].max())).sum())
  mean=sp.mean(0);np.testing.assert_array_equal(new,.75*r003+.25*mean);assert fv.rmse(y,new)==report['scores']['primary']['rmse'];assert fv.rmse(y,mean)==report['scores']['mixup_mlp']['rmse'];assert float(np.sqrt(np.mean(np.array(train)**2)))==report['mean3_train_rmse_pooled'];c=report['comparison'];assert c['gain']==fv.rmse(y,r007)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];assert fg==c['fold_gains'];ix=np.random.default_rng(20270101).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));assert float(boot.std())==c['bootstrap_sd'];np.testing.assert_array_equal(np.percentile(boot,[2.5,97.5]),c['conditional_ci95']);tail={str(b):{'n':int((bins==b).sum()),'reference007_rmse':fv.rmse(y[bins==b],r007[bins==b]),'new_rmse':fv.rmse(y[bins==b],new[bins==b]),'reference_signed_error':float(np.mean(r007[bins==b]-y[bins==b])),'new_signed_error':float(np.mean(new[bins==b]-y[bins==b]))} for b in range(3)};results[split]={'scores':report['scores'],'comparison':c,'train_rmse':report['mean3_train_rmse_pooled'],'reload_max_error':max(errors),'prediction_difference_rms':float(np.sqrt(np.mean((new-r007)**2))),'standalone_validation_outside_foldtrain_label_range':outside,'foldtrain_target_quantile_diagnostics':tail}
 sel=json.loads((OUT/'selection.json').read_text());c=results['original']['comparison'];assert sel['advance']==bool(c['gain']>c['bootstrap_sd'] and c['positive_folds']>=3)
 if not sel['advance']:assert set(results)=={'original'}
 e.check();v={'passed':True,'results':results,'models_verified':count,'independent100epoch_replay_error':replay_error,'protected_files':len(plan['protected']),'seconds':time.perf_counter()-start,'scope':'Development400 only; noholdout/test/candidateCSV/upload/install/cost. Tailbins diagnostic only,fromeachfoldtrain20/80quantiles; no sampleweighting.'};fv.write(_repo_path(ROOT, 'reports/mixup-061-validation.json'),v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
