import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
from common import np,R,O,PLAN,SPLITS
c.check();v=c.J(O/'validation.json');assert v['passed'];z=c.Z(O/'ensemble-OOF.npz');assert c.H(O/'ensemble-OOF.npz')==v['ensemble_sha256']
y=z['y'];P=z['primary'];Q=z['reference'];assert P.shape==Q.shape==(3,400) and len(set(z['groups']))==400
E=(P-y)**2;F=(Q-y)**2;draw=np.random.default_rng(202610203).integers(0,400,(10000,400),dtype=np.int64)
c.arrays(O/'bootstrap-draws.npz',draws=draw)
def influence(a,b):
 benefit=a-b;idx=np.argsort(-benefit,kind='stable');net=float(benefit.sum());res={}
 for n in [1,5,10]:
  keep=np.ones(400,bool);keep[idx[:n]]=False
  res[str(n)]={'benefit_sum':float(benefit[idx[:n]].sum()),'net_fraction':float(benefit[idx[:n]].sum()/net) if net>0 else None,'gain_after_removal':float(np.sqrt(a[keep].mean())-np.sqrt(b[keep].mean()))}
 return {'positive_IDs':int((benefit>0).sum()),'net_SSE_gain':net,'top_benefit_removal':res}
pooledgain=float(np.sqrt(F.mean())-np.sqrt(E.mean()));boot=np.sqrt(F.mean(0)[draw].mean(1))-np.sqrt(E.mean(0)[draw].mean(1));by={}
for si,s in enumerate(SPLITS):
 fg=[float(np.sqrt(F[si,z['fold'][si]==k].mean())-np.sqrt(E[si,z['fold'][si]==k].mean())) for k in range(5)]
 bg=np.sqrt(F[si][draw].mean(1))-np.sqrt(E[si][draw].mean(1))
 by[s]={'reference_RMSE':float(np.sqrt(F[si].mean())),'primary_RMSE':float(np.sqrt(E[si].mean())),'gain':float(np.sqrt(F[si].mean())-np.sqrt(E[si].mean())),'fold_gains':fg,'positive_folds':sum(x>0 for x in fg),'conditional_CI95':np.quantile(bg,[.025,.975]).tolist(),'influence':influence(F[si],E[si])}
vp=P.var(0);vq=Q.var(0);vgain=vq-vp
variance={'partition_prediction_variance_reference':float(vq.mean()),'partition_prediction_variance_primary':float(vp.mean()),'partition_variance_gain':float(vgain.mean()),'conditional_variance_gain_CI95':np.quantile(vgain[draw].mean(1),[.025,.975]).tolist(),'scope':'Across3overlappingpartitions,descriptive training-membership sensitivity;not independentreplications.'}
seed={};ident=[]
for si,s in enumerate(SPLITS):
 row={};coef=z['coefficient'][si]
 for name,arr in [('old3',z['old_seeds'][si]),('new3',z['new_seeds'][si]),('all6',z['all_seeds'][si])]:
  n=arr.shape[0];mean=arr.mean(0);pv=arr.var(0,ddof=0);sv=arr.var(0,ddof=1)
  individual=Q[si][None,:]+coef[None,:]*(arr-z['oldmean'][si][None,:]);avg=individual.mean(0)
  lhs=float(np.mean((individual-y)**2)-np.mean((avg-y)**2));rhs=float(np.mean(coef**2*pv));assert abs(lhs-rhs)<1e-9
  row[name]={'n':n,'raw_mean_RMSE':float(np.sqrt(np.mean((mean-y)**2))),'raw_individual_RMSE':[float(np.sqrt(np.mean((p-y)**2))) for p in arr],'population_seed_variance':float(pv.mean()),'sample_seed_variance':float(sv.mean()),'conditional_raw_mean_variance_proxy':float((sv/n).mean()),'conditional_final_direct_mean_variance_proxy':float((coef**2*sv/n).mean()),'exact_diversity_identity':{'mean_individual_MSE_minus_mean_prediction_MSE':lhs,'weighted_population_variance':rhs,'abs_error':abs(lhs-rhs)}}
  if name=='all6':np.testing.assert_allclose(avg,P[si],atol=1e-10,rtol=0)
 old=z['oldmean'][si];new=z['new_seeds'][si].mean(0);six=z['mean6'][si]
 lhs=.5*np.mean((old-y)**2)+.5*np.mean((new-y)**2)-np.mean((six-y)**2);rhs=np.mean((old-new)**2)/4;assert abs(lhs-rhs)<1e-9
 row['subgroup_identity']={'half_mean_group_MSE_minus_mean6_MSE':float(lhs),'quarter_mean_group_delta_square':float(rhs),'not_gain_vs_old3_alone':True};seed[s]=row
inf=influence(F.mean(0),E.mean(0));ci=np.quantile(boot,[.025,.975]).tolist();delta=P-Q
change={'cross_term':float(np.mean(2*(Q-y)*delta)),'delta_square_term':float(np.mean(delta**2)),'direct_MSE_change':float(np.mean(E-F))};assert abs(change['cross_term']+change['delta_square_term']-change['direct_MSE_change'])<1e-10
criteria={'technical_validation':True,'pooled_gain_positive':pooledgain>0,'all3_gains_nonnegative':all(t['gain']>=0 for t in by.values()),'conditional_CI_lower_positive':ci[0]>0,'pooled_top5_removal_positive':inf['top_benefit_removal']['5']['gain_after_removal']>0,'acrosspartition_variance_not_increased':bool(vgain.mean()>=0)}
summary={'study':203,'by_partition':by,'pooled':{'reference_RMSE':float(np.sqrt(F.mean())),'primary_RMSE':float(np.sqrt(E.mean())),'gain':pooledgain,'conditional_CI95':ci,'influence':inf},'variance':variance,'seed_variance':seed,'exact_MSE_change':change,'criteria':criteria,'strong_development_gate':all(criteria.values()),'primary':'allold3plusallnew3equallyaveraged;frozendownstream','new_seeds':PLAN['new_seeds'],'old_seeds':PLAN['old_seeds'],'new_main_fits':45,'old_training_replays':0,'no_seed_selection':True,'same400_clustered_across3partitions':True,'uncertainty':'Conditionalonfixedfits;historicaladaptivechoice,trainingrefit,correlatedlearners andunknownfamilydependence notcorrected. Seedvarianceproxyusesonly3or6draws;notguaranteedriskgain. Frozenpostprocessingdependsonold3.', 'full500_test_holdout_CSV_upload':False,'cost':0,'plan_sha256':c.H(O/'plan.json'),'ensemble_sha256':c.H(O/'ensemble-OOF.npz')}
c.arrays(O/'bootstrap-results.npz',gain=boot,variance_gain=vgain[draw].mean(1));c.W(O/'summary.json',summary);c.access_report('scoring-access-audit.json');c.emit({'study':203,'pooled':summary['pooled'],'by_partition':{k:{n:v[n] for n in ['reference_RMSE','primary_RMSE','gain','positive_folds']} for k,v in by.items()},'variance':variance,'criteria':criteria,'strong_gate':summary['strong_development_gate']})
