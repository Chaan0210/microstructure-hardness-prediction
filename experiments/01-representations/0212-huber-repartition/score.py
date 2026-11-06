import common as c
from common import R,P,np
c.check();assert c.read(R/'validation.json')['passed'];splits=['original']+c.SPLITS;zs={};old=c.read(P/'experiments/huber-loss-211/summary.json')
with c.Monitor('score',300):
 for split in splits:
  if split=='original':
   z=c.load(P/'experiments/huber-loss-211/OOF.npz');ref=c.load(P/'experiments/huber-loss-211/frozen-reference.npz');saved=c.load(P/'experiments/huber-loss-211/ensemble-OOF.npz')
  else:z=c.load(R/split/'OOF.npz');ref=c.load(R/split/'frozen-reference.npz')
  np.testing.assert_array_equal(z['ids'],ref['ids']);np.testing.assert_array_equal(z['control'],ref['old_first_CNN']);z['fixed_control']=ref['best021']+ref['coefficient']*(z['control']-ref['old_first_CNN']);z['fixed_huber']=ref['best021']+ref['coefficient']*(z['huber']-ref['old_first_CNN']);np.testing.assert_array_equal(z['fixed_control'],ref['best021'])
  if split=='original':
   for name in ['control','huber','fixed_control','fixed_huber']:np.testing.assert_array_equal(z[name],saved[name])
  else:c.arrays(R/split/'ensemble-OOF.npz',ids=z['ids'],y=z['y'],fold=z['fold'],control=z['control'],huber=z['huber'],fixed_control=z['fixed_control'],fixed_huber=z['fixed_huber'],coefficient=ref['coefficient'])
  if zs:
   for key in ['ids','y','groups']:np.testing.assert_array_equal(z[key],zs['original'][key])
  zs[split]=z
 ids=zs['original']['ids'];draws=np.random.default_rng(c.plan['statistics']['seed']).integers(0,400,size=(10000,400));np.testing.assert_array_equal(draws,c.load(P/'experiments/huber-loss-211/bootstrap.npz',['draws'])['draws']);counts=np.stack([np.bincount(row,minlength=400) for row in draws]);boots={};results={};aggregates={};checks=[]
 def compare(ae,be,tag,fold=None,reference_exclude=None):
  assert ae.shape==be.shape and ae.shape[-1]==400
  A=ae.mean(axis=0) if ae.ndim==2 else ae;B=be.mean(axis=0) if be.ndim==2 else be;a=float(np.sqrt(A.mean()));b=float(np.sqrt(B.mean()));boot=np.sqrt(counts@A/400)-np.sqrt(counts@B/400);boots[tag]=boot
  direct=np.array([np.sqrt(ae[...,ix].mean())-np.sqrt(be[...,ix].mean()) for ix in draws[:23]]);assert abs(direct-boot[:23]).max()<1e-12;assert abs(np.sqrt(ae.mean())-np.sqrt(be.mean())-(a-b))<1e-12;order=np.argsort(A-B,kind='stable');keep=order[:-5]
  out={'baseline_RMSE':a,'candidate_RMSE':b,'gain':a-b,'conditional_CI95':np.quantile(boot,[.025,.975]).tolist(),'top5_removed_gain':float(np.sqrt(A[keep].mean())-np.sqrt(B[keep].mean())),'top5_removed_IDs':ids[order[-5:]].tolist(),'positive_ID_count':int(np.sum(A>B)),'ID_count':400,'partition_repetitions':int(ae.shape[0]) if ae.ndim==2 else 1}
  if fold is not None:
   out['fold_results']=[{'fold':k,'baseline_RMSE':float(np.sqrt(A[fold==k].mean())),'candidate_RMSE':float(np.sqrt(B[fold==k].mean())),'gain':float(np.sqrt(A[fold==k].mean())-np.sqrt(B[fold==k].mean()))} for k in range(5)];out['positive_folds']=sum(x['gain']>0 for x in out['fold_results'])
  if reference_exclude is not None:
   keep=~np.isin(ids,reference_exclude);assert keep.sum()==395;out['original211_top5_excluded_gain']=float(np.sqrt(A[keep].mean())-np.sqrt(B[keep].mean()));out['original211_top5_excluded_IDs']=reference_exclude
  checks.append({'comparison':tag,'independent23_draws_passed':True,'squared_error_aggregation_verified':True});return out
 comparisons=[('standalone','control','huber'),('fixed_ensemble','fixed_control','fixed_huber')]
 for split,z in zs.items():
  results[split]={}
  for name,a,b in comparisons:
   q=compare((z['y']-z[a])**2,(z['y']-z[b])**2,split+'_'+name,z['fold'],old['comparisons'][name]['top5_removed_IDs']);q['prediction_delta_RMS']=float(np.sqrt(np.mean((z[b]-z[a])**2)));results[split][name]=q
   if split=='original':
    for key in ['baseline_RMSE','candidate_RMSE','gain','conditional_CI95','top5_removed_gain','top5_removed_IDs','positive_folds']:assert q[key]==old['comparisons'][name][key],key
 for label,parts in [('two_repartitions',c.SPLITS),('all_three_partitions',splits)]:
  aggregates[label]={}
  for name,a,b in comparisons:aggregates[label][name]=compare(np.stack([(zs[s]['y']-zs[s][a])**2 for s in parts]),np.stack([(zs[s]['y']-zs[s][b])**2 for s in parts]),label+'_'+name,reference_exclude=old['comparisons'][name]['top5_removed_IDs'])
 train={'original':old['train_validation']};optimization={'original':old['optimization_diagnostics']};foldrows=[]
 for split in c.SPLITS:
  train[split]={};optimization[split]={};y=zs[split]['y']
  for arm in ['control','huber']:
   errors=[];vals=[]
   for k in range(5):
    pr=c.load(R/split/f'fold-{k}-{arm}/predictions.npz');er=(y[pr['train_indices']]-pr['train_predictions'])**2;errors.extend(er.tolist());vals.append(c.load(R/split/f'fold-{k}-{arm}/streams.npz')['values']);foldrows.append({'split':split,'fold':k,'arm':arm,'train_RMSE':float(np.sqrt(er.mean())),'OOF_RMSE':float(np.sqrt(np.mean((y[pr['validation_indices']]-pr['validation_predictions'])**2)))})
   v=np.concatenate(vals);train[split][arm]={'pooled_train_RMSE_over1600_repeated_train_observations':float(np.sqrt(np.mean(errors))),'OOF_RMSE':results[split]['standalone']['baseline_RMSE' if arm=='control' else 'candidate_RMSE']};train[split][arm]['OOF_minus_train_gap']=train[split][arm]['OOF_RMSE']-train[split][arm]['pooled_train_RMSE_over1600_repeated_train_observations'];optimization[split][arm]={'updates':4000,'mean_optimized_loss':float(v[:,0].mean()),'mean_unclipped_gradient_norm':float(v[:,1].mean()),'clip5_fraction':float(np.mean(v[:,1]>5)),'mean_unweighted_MSE':float(v[:,2].mean()),'mean_scaled_Huber':float(v[:,3].mean()),'mean_tail_fraction_abs_r_gt1':float(v[:,4].mean())}
 gates={}
 for split in c.SPLITS:
  q=results[split]['standalone'];gates[split]={'gain_at_least_0_02':q['gain']>=.02,'at_least3_positive_folds':q['positive_folds']>=3,'CI_lower_positive':q['conditional_CI95'][0]>0,'top5_removed_positive':q['top5_removed_gain']>0}
 for label,z in aggregates.items():
  q=z['standalone'];gates[label]={'gain_positive':q['gain']>0,'CI_lower_positive':q['conditional_CI95'][0]>0,'top5_removed_positive':q['top5_removed_gain']>0}
 eligible=all(all(v.values()) for v in gates.values());c.arrays(R/'bootstrap.npz',draws=draws,**boots)
 summary={'study':212,'status':'고정 손실 정책의 두 재분할 안정성 대조 완료','loss_policy':c.plan['loss_policy'],'individual_partitions':results,'ID_cluster_aggregates':aggregates,'train_validation':train,'fold_train_validation':foldrows,'optimization_diagnostics':optimization,'stability_criteria':gates,'stability_criteria_passed':eligible,'decision':'추가 재분할 및 pooled의 사전 안정성 기준을 통과했다. 원본 211의 CI 기준 미달 판단은 유지하며 보고 후 정지한다.' if eligible else '사전 분할 안정성 기준을 충족하지 못했다. 원본 211의 CI 기준 미달 판단과 best021을 유지하고 보고 후 정지한다.','original211_decision':old['decision'],'original211_criteria':old['criteria'],'original211_point_estimates_CIs_top5_exact':True,'original211_retrained':False,'best021_replaced':False,'main_gradient_fits':20,'verification_gradient_fits':4,'new_gradient_updates':19200,'loss_policy_count':1,'fixed_ensemble_meaning':c.plan['fixed_ensemble'],'statistics':c.plan['statistics'],'gain_definition':'MSE 대조 RMSE - 고정 Huber RMSE. 양수가 개선.','limitations':c.plan['limitations'],'plan_SHA256':c.sha(R/'plan.json'),'validation_SHA256':c.sha(R/'validation.json'),'bootstrap_SHA256':c.sha(R/'bootstrap.npz'),'cost':0,'install_download_holdout_test_full500_CSV_submission':False}
 c.write(R/'score-validation.json',{'passed':True,'checks':checks,'same400_IDs_labels_groups_exact':True,'original211_points_CIs_top5_exact':True,'original211_draws_exact':True,'draw_unit':'400 ID clusters','independent1200_bootstrap':False});c.write(R/'summary.json',summary);c.emit({'decision':summary['decision'],'individual_partitions':results,'aggregates':aggregates,'criteria':gates,'train_validation':train})
c.check()
