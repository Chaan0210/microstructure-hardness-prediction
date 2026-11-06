import common as c
from common import R,P,np
c.check();assert c.read(R/'validation.json')['passed'];info=c.read(R/'training-complete.json')['splits'];splits=['original']+c.SPLITS;zs={};refs={};old=c.read(P/'experiments/resolution-216/summary.json')
with c.Monitor('score',600):
 for split in splits:
  root=P/'experiments/resolution-216' if split=='original' else R/split;z=c.load(root/'OOF.npz');ref=c.load(root/'frozen-reference.npz');np.testing.assert_array_equal(z['ids'],ref['ids']);z['fixed_control']=ref['best021']+ref['coefficient']*(z['control']-ref['old_first_CNN']);z['fixed_native']=ref['best021']+ref['coefficient']*(z['native']-ref['old_first_CNN']);z['best021']=ref['best021']
  if split=='original' or info[split]['reused_controls']:np.testing.assert_array_equal(z['control'],ref['old_first_CNN']);np.testing.assert_array_equal(z['fixed_control'],ref['best021'])
  if zs:
   for key in ['ids','y','groups']:np.testing.assert_array_equal(z[key],zs['original'][key])
  if split!='original':c.arrays(root/'ensemble-OOF.npz',ids=z['ids'],y=z['y'],fold=z['fold'],control=z['control'],native=z['native'],fixed_control=z['fixed_control'],fixed_native=z['fixed_native'],best021=z['best021'],coefficient=ref['coefficient'])
  zs[split]=z;refs[split]=ref
 ids=zs['original']['ids'];draws=np.random.default_rng(c.plan['statistics']['seed']).integers(0,400,size=(10000,400));np.testing.assert_array_equal(draws,c.load(P/'experiments/resolution-216/bootstrap.npz',['draws'])['draws']);counts=np.stack([np.bincount(row,minlength=400) for row in draws]);boots={};results={};aggregates={};errors={};diagnostics={}
 def compare(ae,be,tag,fold=None,excluded=None):
  assert ae.shape==be.shape and ae.shape[-1]==400;A=ae.mean(0) if ae.ndim==2 else ae;B=be.mean(0) if be.ndim==2 else be;boot=np.sqrt(counts@A/400)-np.sqrt(counts@B/400);boots[tag]=boot;errors[tag+'_A']=ae;errors[tag+'_B']=be;idx=np.argsort(A-B,kind='stable');keep=idx[:-5];r={'baseline_RMSE':float(np.sqrt(A.mean())),'candidate_RMSE':float(np.sqrt(B.mean())),'gain':float(np.sqrt(A.mean())-np.sqrt(B.mean())),'conditional_CI95':np.quantile(boot,[.025,.975]).tolist(),'top5_removed_gain':float(np.sqrt(A[keep].mean())-np.sqrt(B[keep].mean())),'top5_removed_IDs':ids[idx[-5:]].tolist(),'positive_IDs':int(np.sum(A>B)),'ID_count':400,'repetitions':ae.shape[0] if ae.ndim==2 else 1}
  direct=np.array([np.sqrt(ae[...,ix].mean())-np.sqrt(be[...,ix].mean()) for ix in draws[:23]]);np.testing.assert_allclose(direct,boot[:23],rtol=0,atol=1e-12)
  if fold is not None:r['fold_gains']=[float(np.sqrt(A[fold==k].mean())-np.sqrt(B[fold==k].mean())) for k in range(5)];r['positive_folds']=sum(x>0 for x in r['fold_gains'])
  if excluded is not None:
   keep=~np.isin(ids,excluded);assert keep.sum()==395;r['fixed216_top5_excluded_gain']=float(np.sqrt(A[keep].mean())-np.sqrt(B[keep].mean()));r['fixed216_top5_excluded_IDs']=excluded
  return r
 comparisons=[('standalone','control','native'),('fixed_ensemble','fixed_control','fixed_native'),('versus_best021','best021','fixed_native')]
 for split,z in zs.items():
  results[split]={};rows=[]
  for name,a,b in comparisons:
   q=compare((z['y']-z[a])**2,(z['y']-z[b])**2,split+'_'+name,z['fold'],old['comparisons'][name]['top5_removed_IDs']);results[split][name]=q
   if split=='original':
    orig=old['comparisons'][name]
    for key in ['baseline_RMSE','candidate_RMSE','conditional_CI95','top5_removed_gain','fold_gains']:np.testing.assert_allclose(q[key],orig[key],rtol=0,atol=1e-12)
    assert abs(q['gain']-orig['gain_baseline_minus_candidate'])<1e-12;assert q['top5_removed_IDs']==orig['top5_removed_IDs']
  for i,ID in enumerate(ids):rows.append({'ID':str(ID),'fold':int(z['fold'][i]),'target':float(z['y'][i]),**{n:float(z[n][i]) for n in ['control','native','fixed_control','fixed_native','best021']},**{n+'_SEgain':float((z['y'][i]-z[a][i])**2-(z['y'][i]-z[b][i])**2) for n,a,b in comparisons}})
  if split!='original':c.write(R/split/'ID-residuals.json',rows)
  # Algebraic complementarity diagnostics only, no fitted coefficients or tuned blends.
  ref=refs[split];w=ref['coefficient'];a=z['control'];b=z['native'];y=z['y'];pa=z['fixed_control'];pb=z['fixed_native'];delta=pb-pa;ra=pa-y;ca=a-y;cb=b-y;remainder=(pa-w*a)/(1-w);rr=remainder-y;seeds=c.load(P/f'artifacts/cnn-seed-mean-133/audit/{split}.npz');np.testing.assert_array_equal(seeds['ids'],ids);np.testing.assert_array_equal(seeds['seeds'][0],ref['old_first_CNN']);other=seeds['seeds'][1:].mean(0)-y
  corr=lambda x,t:float(np.corrcoef(x,t)[0,1]);alignment=float(-2*np.mean(ra*delta));penalty=float(np.mean(delta**2));gain=float(np.mean((pa-y)**2-(pb-y)**2));assert abs(alignment-penalty-gain)<1e-12
  soloalign=float(-2*np.mean(ca*(b-a)));solopen=float(np.mean((b-a)**2));assert abs(soloalign-solopen-np.mean(ca**2-cb**2))<1e-12
  diagnostics[split]={'CNN128_vs256_residual_correlation':corr(ca,cb),'CNN128_vs_fixed_baseline_residual_correlation':corr(ca,ra),'CNN256_vs_fixed_baseline_residual_correlation':corr(cb,ra),'CNN128_vs_removed_firstseed_remainder_residual_correlation':corr(ca,rr),'CNN256_vs_removed_firstseed_remainder_residual_correlation':corr(cb,rr),'CNN128_vs_other_two_seed_mean_residual_correlation':corr(ca,other),'CNN256_vs_other_two_seed_mean_residual_correlation':corr(cb,other),'replacement_delta_vs_baseline_residual_correlation':corr(delta,ra),'fixed_MSE_gain_decomposition':{'alignment_minus2mean_r_delta':alignment,'delta_squared_penalty':penalty,'net_MSEgain':gain},'standalone_MSE_gain_decomposition':{'alignment':soloalign,'delta_squared_penalty':solopen,'net_MSEgain':float(np.mean(ca**2-cb**2))},'mean_residual128':float(ca.mean()),'mean_residual256':float(cb.mean()),'delta_prediction_RMS':float(np.sqrt(np.mean((b-a)**2))),'fixed_delta_RMS':float(np.sqrt(np.mean(delta**2)))}
 for name,parts in [('two_repartitions',c.SPLITS),('all_three_partitions',splits)]:
  aggregates[name]={}
  for label,a,b in comparisons:aggregates[name][label]=compare(np.stack([(zs[t]['y']-zs[t][a])**2 for t in parts]),np.stack([(zs[t]['y']-zs[t][b])**2 for t in parts]),name+'_'+label,excluded=old['comparisons'][label]['top5_removed_IDs'])
 def gate(q,pooled=False):return {'gain_threshold':q['gain']>(0 if pooled else -.000000000000001) if pooled else q['gain']>=.02,'CI_lower_positive':q['conditional_CI95'][0]>0,'top5_removed_positive':q['top5_removed_gain']>0,**({} if pooled else {'at_least3_positive_folds':q['positive_folds']>=3})}
 gates={name:{split:gate(results[split][name]) for split in c.SPLITS} for name,_,_ in comparisons}
 for name,_,_ in comparisons:gates[name]['two_repartitions']=gate(aggregates['two_repartitions'][name],True)
 passed={name:all(all(v.values()) for v in checks.values()) for name,checks in gates.items()}
 # A downstream refit may test a different pipeline but is not justified as an automatic rescue.
 decision='단독 분할안정성 기준 통과.' if passed['standalone'] else '단독 분할안정성 기준 미달.'
 downstream='고정 치환도 안정성 기준을 통과해 향후 별도승인된 누출없는 후단 재학습 비교를 검토할 근거가 있다.' if passed['standalone'] and passed['fixed_ensemble'] and passed['versus_best021'] else '현재 근거로 전체 후단 재학습을 다음 우선 실험으로 권하지 않는다. 후단 적응이 효과를 회복한다는 근거는 없고, 실제 전체 파이프라인 효과를 주장하려면 별도승인된 nested 재학습 대조가 필요하다.'
 c.arrays(R/'bootstrap.npz',draws=draws,**boots);c.arrays(R/'comparison-squared-errors.npz',ids=ids,**errors);c.write(R/'complementarity-diagnostics.json',{'partitions':diagnostics,'meaning':'Residual correlations and exact MSE identities; algebraic remainder holds fitted downstream fixed, not an independently trained non-CNN model; no OOF weight optimization','downstream_assessment':downstream})
 c.write(R/'summary.json',{'study':217,'individual_partitions':results,'ID_cluster_aggregates':aggregates,'stability_criteria':gates,'stability_passed':passed,'decision':decision,'downstream_assessment':downstream,'original216_standalone_gate_preserved':old['criteria'],'original216_ensemble_further_review_eligible_preserved':old['ensemble_further_review_eligible'],'original216_retrained':False,'best021_replaced':False,'statistics':c.plan['statistics'],'limitations':c.plan['limitations'],'no_downstream_retraining':True,'cost':0,'no_install_download_holdout_test_full500_CSV_submission':True});c.emit({'decision':decision,'individual_partitions':results,'aggregates':aggregates,'stability_passed':passed,'downstream_assessment':downstream})
c.check()
