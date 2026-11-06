import common as c,engine as e
from common import R,C,np,torch
c.check();assert c.device=='mps';X,cache,d=c.inputs();records=[]
with c.Monitor('preflight',300):
 original=c.load(C/'convnext_tiny/adapt-OOF.npz')
 for key,name in [('ids','ids'),('y','y'),('fold','fold'),('groups','g')]:np.testing.assert_array_equal(original[key],d[name])
 for k in range(5):
  T,V=c.indices(k);folder=C/'convnext_tiny'/f'fold-{k}';warm=e.ck(folder/'initial-head.pt');expected=e.floor_stats(X,d['y'],T,V,202710185+k);poison=d['y'].copy();poison[V]=np.nan;ps=e.floor_stats(X,poison,T,V,202710185+k)
  for key in ['feature_mean','feature_var','feature_scale']:assert torch.equal(warm[key],expected[key]) and torch.equal(warm[key],ps[key])
  for key in ['train','validation','seed','dim','target_mean','target_std','feature_RMS_floor']:assert warm[key]==expected[key]==ps[key]
  obj=e.ck(folder/'last_stage112/model.pt');prefix,tail=c.model('convnext_tiny');assert c.statehash(prefix)==obj['prefix_initial'] and c.statehash(tail)==obj['tail_initial'];del prefix;tail.load_state_dict(obj['tail']);tail.to('mps');h=e.old.head(768);h.load_state_dict(obj['head']);h.to('mps');pred=e.old.predict(h,tail,cache,X,V,obj,'mps');np.testing.assert_array_equal(pred,original['last_stage112'][V]);trainpred=e.old.predict(h,tail,cache,X,T,obj,'mps');trainrmse=float(np.sqrt(np.mean((d['y'][T]-trainpred)**2)));assert trainrmse==c.read(folder/'last_stage112/report.json')['final_train_RMSE'];tail.requires_grad_(True);groups,mapping=e.parameter_groups(h,tail,.8);assert [r['initial_lr'] for r in mapping]==c.plan['initial_LRs'];assert c.rnghash()==c.rnghash();records.append({'fold':k,'OOF_MPS_reload_exact':True,'train_RMSE_exact':True,'normalization_train_only_poison_exact':True,'official_prefix_initial_tail_hashes_exact':True,'parameter_groups':mapping});c.emit({'preflight_fold':k,'passed':True});del tail,h,obj,groups;__import__('gc').collect();torch.mps.empty_cache()
 c.write(R/'decay-unit.json',e.decay_unit());c.write(R/'preflight.json',{'passed':True,'records':records,'historical_per_update_RNG_available':False,'RNG_comparison_method':c.plan['RNG_validation'],'gradient_fits':0,'candidate_scores_computed':False,'plan_SHA256':c.sha(R/'plan.json')})
c.check()
