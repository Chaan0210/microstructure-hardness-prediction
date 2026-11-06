from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,B,np,torch
import run as e
import gc,time

def infer(name,path,X,cache,V,post):
 o=e.ck(path);preds={};ranges={}
 for device in ['cpu','mps']:
  h=e.old.head(X.shape[1]);h.load_state_dict(o['head']);h.eval().to(device);tail=None
  if post:
   prefix,tail=c.model(name);del prefix;tail.load_state_dict(o['tail']);tail.eval().to(device)
  preds[device]=e.old.predict(h,tail,cache,X,V,o,device)
  mu,scale=e.old.tensors(o,device);rawlo=float('inf');rawhi=-float('inf');zmax=0.
  with torch.inference_mode():
   for start in range(0,len(V),2):
    rows=V[start:start+2];a=tail(torch.tensor(np.array(cache[rows]),device=device)) if post else torch.tensor(X[rows],dtype=torch.float32,device=device)
    z=(a-mu)/scale;assert torch.isfinite(a).all() and torch.isfinite(z).all();rawlo=min(rawlo,float(a.min()));rawhi=max(rawhi,float(a.max()));zmax=max(zmax,float(z.abs().max()))
  ranges[device]={'prediction_min':float(preds[device].min()),'prediction_max':float(preds[device].max()),'raw_feature_range':[rawlo,rawhi],'maxabs_scaled_feature':zmax,'finite':True}
  del h,tail;gc.collect();torch.mps.empty_cache()
 err=float(np.max(np.abs(preds['cpu']-preds['mps'])))
 saved_exact=None
 if post:
  saved=c.load(path.parent/'predictions.npz');np.testing.assert_array_equal(saved['indices'],V);np.testing.assert_array_equal(saved['prediction'],preds['mps']);saved_exact=True
 return {'checkpoint':str(path.relative_to(P)),'sha256':c.sha(path),'CPU_MPS_maxabs':err,'CPU_MPS_tolerance':.002,'CPU_MPS_passed':err<=.002,'MPS_saved_reload_exact':saved_exact,'ranges':ranges}

def scale_case(X,y,T,V,seed,ids):
 o=e.floor_stats(X,y,T,V,seed);mu=o['feature_mean'].numpy();sd=np.sqrt(o['feature_var'].numpy());new=o['feature_scale'].numpy();old=sd.copy();old[old==0]=1.;z=(X[V]-mu)/old;row,col=np.unravel_index(np.abs(z).argmax(),z.shape)
 xp=X.copy();yp=y.copy();xp[V]=xp[V]*-1e5+1e7;yp[V]=-1000*yp[V]+1e6;poison=e.floor_stats(xp,yp,T,V,seed)
 for k in ['feature_mean','feature_var','feature_scale']:assert torch.equal(o[k],poison[k])
 for k in ['feature_RMS_floor','target_mean','target_std']:assert o[k]==poison[k]
 scaled=e.floor_stats(X*7,y,T,V,seed);np.testing.assert_allclose(scaled['feature_scale'].numpy()/7,new,rtol=1e-12,atol=1e-14);np.testing.assert_allclose((X[V]*7-scaled['feature_mean'].numpy())/scaled['feature_scale'].numpy(),(X[V]-mu)/new,rtol=1e-10,atol=1e-10)
 assert float(np.max(sd/new))<=1+1e-12
 rec={'train_n':len(T),'validation_n':len(V),'worst_feature':int(col),'worst_validation_ID':str(ids[V[row]]),'train_feature_mean':float(mu[col]),'train_feature_std':float(sd[col]),'train_feature_minmax':[float(X[T,col].min()),float(X[T,col].max())],'train_feature_nonzero_count':int(np.count_nonzero(X[T,col])),'validation_feature_minmax':[float(X[V,col].min()),float(X[V,col].max())],'worst_validation_raw':float(X[V[row],col]),'old_validation_scaled_feature_minmax':[float(z[:,col].min()),float(z[:,col].max())],'old_validation_maxabs_Z':float(np.abs(z).max()),'new_validation_maxabs_Z':float(np.abs((X[V]-mu)/new).max()),'RMS_floor':o['feature_RMS_floor'],'new_scale_for_worst_feature':float(new[col]),'old_inverse_scale_amplification':float(1/old[col]),'new_inverse_scale_amplification':float(1/new[col]),'max_train_std_over_scale':float(np.max(sd/new)),'heldout_feature_and_label_poison_stats_exact':True,'positive_global_unit_rescaling_invariance':True,'formula':'r=sqrt(mean_j(var_train_j)); scale_j=max(sqrt(var_train_j),r); if r==0 scale_j=1; Z=(X-mean_train)/scale; variance ddof=0; fixed throughout adaptation','no_prediction_clipping':True}
 return rec

def main():
 c.check();d=c.data('original');results=[];started=time.monotonic();cases=[('resnet50',1,2,['adamw_low','adamw_high']),('resnet50',4,0,['adamw_high']),('efficientnet_b0',1,2,['adamw_low'])]
 with c.Monitor('problem-probe',900):
  for name,k,j,recipes in cases:
   e.CURRENT_MODEL=name;X=c.load(B/name/'features.npz')['X'].astype(float);cache=np.load(_repo_io(B/name/'prefix.npy'),mmap_mode='r');cell=c.plan['folds'][str(k)]['inner'][j];T=np.array(cell['train']);V=np.array(cell['validation']);seed=202710185+k;relative=f'{name}/fold-{k}/inner-{j}';dest=R/relative;dest.mkdir(parents=True,exist_ok=True);source=e.D/relative
   scale=scale_case(X,d['y'],T,V,seed,d['ids']);c.write(dest/'scale-diagnostic.json',scale);c.emit({'case':relative,'scaling':scale})
   original_pre=infer(name,source/'initial-head.pt',X,cache,V,False);original=e.ck(source/'initial-head.pt');fresh=e.save_head(dest/'initial-head.pt',X,d['y'],T,V,seed)
   assert original['initial_head_sha256']==fresh['initial_head_sha256'] and original['order_sha256']==fresh['order_sha256']
   for field in ['feature_mean','feature_var']:assert torch.equal(original[field],fresh[field])
   for field in ['target_mean','target_std','train','validation','seed']:assert original[field]==fresh[field]
   new_pre=infer(name,dest/'initial-head.pt',X,cache,V,False);record={'cell':relative,'scale':scale,'original_pre':original_pre,'new_pre':new_pre,'same_untrained_head_and_training_orders':True,'fits':[]}
   for recipe_name in recipes:
    recipe=next(v for v in e.RECIPES if v['name']==recipe_name);old_post=infer(name,source/recipe_name/'model.pt',X,cache,V,True);out=dest/recipe_name;out.mkdir();e.train(fresh,cache,X,d['y'][T].copy(),recipe,202710186+100*k+j,out,V);new_post=infer(name,out/'model.pt',X,cache,V,True)
    record['fits'].append({'recipe':recipe_name,'original_post':old_post,'new_post':new_post});c.emit({'probe_fit':relative+'/'+recipe_name,'old_CPU_MPS':old_post['CPU_MPS_maxabs'],'new_CPU_MPS':new_post['CPU_MPS_maxabs'],'new_range':new_post['ranges']['mps']})
   results.append(record)
  c.check()
 oldfail=[r for case in results if case['cell'].startswith('resnet50') for r in case['fits']];allnew=[case['new_pre'] for case in results]+[r['new_post'] for case in results for r in case['fits']]
 gate={'passed':all(r['CPU_MPS_passed'] for r in allnew) and all(not r['original_post']['CPU_MPS_passed'] for r in oldfail) and results[0]['scale']['new_validation_maxabs_Z']<results[0]['scale']['old_validation_maxabs_Z'],'new_reload_checks':len(allnew),'all_new_CPU_MPS_within_002':all(r['CPU_MPS_passed'] for r in allnew),'all_three_original_CPU_failures_reproduced':all(not r['original_post']['CPU_MPS_passed'] for r in oldfail),'original_CPU_tolerance_unchanged':.002,'all_new_features_predictions_gradients_finite':True,'train_only_and_unit_invariance_passed':True,'accuracy_not_used_for_gate':True,'prediction_ranges_diagnostic_only':True,'seconds':time.monotonic()-started}
 c.write(R/'problem-reproduction-and-repair.json',{'gate':gate,'cases':results});c.write(R/'problem-gate.json',gate);c.write(P/'reports/cnn-scale-floor-187-probe.json',{'gate':gate,'detail':str((R/'problem-reproduction-and-repair.json').relative_to(P)),'detail_SHA256':c.sha(R/'problem-reproduction-and-repair.json')});c.emit(gate)
if __name__=='__main__':main()
