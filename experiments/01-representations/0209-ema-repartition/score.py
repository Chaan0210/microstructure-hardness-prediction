import common as c
from common import R,P,np
c.check();assert c.read(R/'validation.json')['passed'];splits=['original']+c.SPLITS;zs={};old208=c.read(P/'experiments/update-ema-208/summary.json')
with c.Monitor('score',300):
 for split in splits:
  if split=='original':z=c.load(P/'experiments/update-ema-208/OOF.npz');ref=c.load(P/'experiments/contrast-augmentation-206/frozen-reference.npz');saved=c.load(P/'experiments/update-ema-208/ensemble-OOF.npz')
  else:z=c.load(R/split/'OOF.npz');ref=c.load(R/split/'frozen-reference.npz')
  np.testing.assert_array_equal(z['ids'],ref['ids']);np.testing.assert_array_equal(z['last5_uniform'],ref['old_first_CNN']);z['fixed_uniform']=ref['best021']+ref['coefficient']*(z['last5_uniform']-ref['old_first_CNN']);z['fixed_ema']=ref['best021']+ref['coefficient']*(z['update_ema']-ref['old_first_CNN']);np.testing.assert_array_equal(z['fixed_uniform'],ref['best021'])
  if split=='original':np.testing.assert_array_equal(z['fixed_uniform'],saved['last5_uniform']);np.testing.assert_array_equal(z['fixed_ema'],saved['update_EMA'])
  else:c.arrays(R/split/'ensemble-OOF.npz',ids=z['ids'],y=z['y'],fold=z['fold'],last5_uniform=z['fixed_uniform'],update_EMA=z['fixed_ema'],coefficient=ref['coefficient'])
  if zs:
   for key in ['ids','y','groups']:np.testing.assert_array_equal(z[key],zs['original'][key])
  zs[split]=z
 ids=zs['original']['ids'];draws=np.random.default_rng(c.plan['statistics']['seed']).integers(0,400,size=(10000,400));counts=np.stack([np.bincount(row,minlength=400) for row in draws]);boots={};results={};aggregate={};validation=[]
 def compare(ae,be,tag,fold=None,reference_exclude=None):
  # The last dimension is always the 400 IDs. Keep partition repetitions together.
  assert ae.shape==be.shape and ae.shape[-1]==400
  A=ae.mean(axis=0) if ae.ndim==2 else ae;B=be.mean(axis=0) if be.ndim==2 else be;a=float(np.sqrt(A.mean()));b=float(np.sqrt(B.mean()));gain=a-b;boot=np.sqrt(counts@A/400)-np.sqrt(counts@B/400);boots[tag]=boot
  direct=np.array([np.sqrt(ae[...,ix].mean())-np.sqrt(be[...,ix].mean()) for ix in draws[:23]]);assert abs(direct-boot[:23]).max()<1e-12;assert abs(np.sqrt(ae.mean())-np.sqrt(be.mean())-gain)<1e-12;order=np.argsort(A-B,kind='stable');keep=order[:-5]
  out={'baseline_RMSE':a,'EMA_RMSE':b,'gain':gain,'conditional_CI95':np.quantile(boot,[.025,.975]).tolist(),'top5_removed_gain':float(np.sqrt(A[keep].mean())-np.sqrt(B[keep].mean())),'top5_removed_IDs':ids[order[-5:]].tolist(),'positive_ID_count':int(np.sum(A>B)),'ID_count':400,'partition_repetitions':int(ae.shape[0]) if ae.ndim==2 else 1}
  if fold is not None:
   out['fold_results']=[{'fold':k,'baseline_RMSE':float(np.sqrt(A[fold==k].mean())),'EMA_RMSE':float(np.sqrt(B[fold==k].mean())),'gain':float(np.sqrt(A[fold==k].mean())-np.sqrt(B[fold==k].mean()))} for k in range(5)];out['positive_folds']=sum(x['gain']>0 for x in out['fold_results'])
  if reference_exclude is not None:
   keep=~np.isin(ids,reference_exclude);assert keep.sum()==395;out['original208_top5_excluded_gain']=float(np.sqrt(A[keep].mean())-np.sqrt(B[keep].mean()));out['original208_top5_excluded_IDs']=reference_exclude
  validation.append({'comparison':tag,'independent_23_draw_recalculation_passed':True,'nested_squares_aggregate_verified':True});return out
 for split,z in zs.items():
  results[split]={}
  for name,a,b in [('standalone','last5_uniform','update_ema'),('fixed_ensemble','fixed_uniform','fixed_ema')]:
   q=compare((z['y']-z[a])**2,(z['y']-z[b])**2,split+'_'+name,z['fold'],old208['comparisons'][name]['top5_removed_IDs']);q['prediction_delta_RMS']=float(np.sqrt(np.mean((z[b]-z[a])**2)));results[split][name]=q
   if split=='original':
    for key in ['baseline_RMSE','EMA_RMSE','gain','top5_removed_gain']:assert q[key]==old208['comparisons'][name][key]
    q['historical208_CI95_seed202610208']=old208['comparisons'][name]['conditional_CI95']
 for label,partitions in [('two_repartitions',c.SPLITS),('all_three_partitions',splits)]:
  aggregate[label]={}
  for name,a,b in [('standalone','last5_uniform','update_ema'),('fixed_ensemble','fixed_uniform','fixed_ema')]:aggregate[label][name]=compare(np.stack([(zs[s]['y']-zs[s][a])**2 for s in partitions]),np.stack([(zs[s]['y']-zs[s][b])**2 for s in partitions]),label+'_'+name,reference_exclude=old208['comparisons'][name]['top5_removed_IDs'])
 gates={}
 for split in c.SPLITS:
  q=results[split]['standalone'];gates[split]={'gain_at_least_0_02':q['gain']>=.02,'at_least_3_positive_folds':q['positive_folds']>=3,'CI_lower_positive':q['conditional_CI95'][0]>0,'top5_removed_positive':q['top5_removed_gain']>0}
 for label,z in aggregate.items():
  q=z['standalone'];gates[label]={'gain_positive':q['gain']>0,'CI_lower_positive':q['conditional_CI95'][0]>0,'top5_removed_positive':q['top5_removed_gain']>0}
 eligible=all(all(v.values()) for v in gates.values());c.arrays(R/'bootstrap.npz',draws=draws,**boots)
 s={'study':209,'status':'기존 두 재분할에서 고정 EMA 정책 검증 완료','policy':c.plan['EMA'],'individual_partitions':results,'ID_cluster_aggregates':aggregate,'stability_criteria':gates,'stability_criteria_passed':eligible,'decision':'CNN 단독의 고정 EMA 분할 안정성 기준을 통과했다. 후단 전체 재학습 여부는 별도 판단하며 이번에는 실행하지 않는다.' if eligible else '고정 EMA의 분할 안정성 기준을 충족하지 못했다. 이번 정책의 확대를 중단하고 best021을 유지한다.','gradient_training_fits':10,'gradient_updates':8000,'shadow_optimizer_verification_steps':8000,'additional_gradient_validation_fits':0,'EMA_policy_count':1,'original208_retrained':False,'best021_replaced':False,'fixed_ensemble_meaning':c.plan['fixed_ensemble'],'statistics':c.plan['statistics'],'gain_definition':'기존 마지막 5 epoch 평균의 RMSE - 고정 전체 update EMA의 RMSE. 양수가 개선이다.','limitations':['세 분할은 같은 개발 400개 ID의 반복측정이다. 독립 holdout 또는 1,200개 독립 표본이 아니다.','ID cluster bootstrap은 저장된 학습 결과에 조건부다. 재학습·seed 변동과 누적 400장 선택 편향은 보정하지 않는다.','고정 앙상블은 기존 CNN 3seed 중 첫 seed 출력만 치환했다. 추가 blend가 아니며 다른 seed·calibrator·잔차는 고정했다. 전체 nested 재학습의 성능을 추정하는 결과로 단정하지 않는다.','201의 과거 update별 RNG 해시는 없다. 시작·종료 RNG, 모든 배치·증강·LR·MSE·main gradient norm, 마지막 5 state와 평균, 최종 main optimizer·scheduler 및 OOF를 직접 비교했다.','EMA decay·warmup·초기화·seed·epoch·평가 시점은 탐색하지 않았다.'], 'plan_SHA256':c.sha(R/'plan.json'),'validation_SHA256':c.sha(R/'validation.json'),'bootstrap_SHA256':c.sha(R/'bootstrap.npz'),'cost':0,'install_download_holdout_test_full500_newCSV_submission':False}
 c.write(R/'score-validation.json',{'passed':True,'comparisons':validation,'same400IDs_labels_groups_exact_all_partitions':True,'original208_point_estimates_unchanged':True,'draw_unit':'400 ID clusters','no_independent1200_bootstrap':True});c.write(R/'summary.json',s);c.emit({'decision':s['decision'],'individual_partitions':results,'aggregates':aggregate,'criteria':gates})
c.check()
