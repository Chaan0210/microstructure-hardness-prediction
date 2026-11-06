import math
import common as c
from common import R,P,np
c.check();r=c.read(R/'report.json');b=c.load(R/'bootstrap.npz');s=c.load(R/'ID-paired-aggregation.npz');old=c.load(P/'experiments/matched-replacement-194/replacement-OOF.npz');raw=c.load(P/'experiments/matched-replacement-194/legacy-precision.npz');parts={'original':{**old,**{'raw_'+a:raw[a] for a in c.ARMS}}}
for part in c.SPLITS:parts[part]=c.load(R/part/'OOF.npz')
np.testing.assert_array_equal(b['draws'],np.random.default_rng(202710201).integers(0,400,(3000,400)));ix=b['draws'];errors={};maxboot=0.
def rmse(y,p):return math.sqrt(math.fsum((float(v)-float(w))**2 for v,w in zip(y,p))/len(y))
for part,d in parts.items():
 np.testing.assert_array_equal(d['ids'],old['ids']);np.testing.assert_array_equal(d['y'],old['y']);errors[part]={};rep=r['individual_partitions'][part]
 for name,q in rep['scores'].items():assert abs(rmse(d['y'],d[name])-q['RMSE'])<1e-12
 for name,q in rep['pairs'].items():
  a=d[q['reference']];v=d[q['candidate']];y=d['y'];ea=(a-y)**2;eb=(v-y)**2;bd=np.sqrt(np.mean(ea[ix],1))-np.sqrt(np.mean(eb[ix],1));err=float(abs(bd-b[part+'__'+name]).max());assert err<1e-12;maxboot=max(maxboot,err);np.testing.assert_allclose(np.quantile(bd,[.025,.975]),q['conditional_IDpaired_bootstrap_CI95'],atol=1e-12,rtol=0);order=sorted(range(400),key=lambda i:(-((float(a[i])-float(y[i]))**2-(float(v[i])-float(y[i]))**2),str(d['ids'][i])));assert d['ids'][order[:5]].tolist()==q['top5_removed_IDs'];keep=order[5:];assert abs(rmse(y[keep],a[keep])-rmse(y[keep],v[keep])-q['top5_removed_gain'])<1e-12;errors[part][name]=(ea,eb)
 for k in range(5):
  m=d['fold']==k
  for name,q in rep['pairs'].items():assert abs(rmse(d['y'][m],d[q['reference']][m])-rmse(d['y'][m],d[q['candidate']][m])-q['fold_gains'][k])<1e-12
for name,q in r['IDpaired_three_partition_diagnostic'].items():
 A=np.stack([errors[p][name][0] for p in parts]);B=np.stack([errors[p][name][1] for p in parts]);a=A.mean(0);v=B.mean(0);np.testing.assert_array_equal(s[name+'__ID_mean_reference_SE'],a);np.testing.assert_array_equal(s[name+'__ID_mean_candidate_SE'],v);np.testing.assert_array_equal(s[name+'__paired_SE_gain_by_partition'],A-B);bd=np.sqrt(a[ix].mean(1))-np.sqrt(v[ix].mean(1));err=float(abs(bd-b['aggregate__'+name]).max());assert err<1e-12;maxboot=max(maxboot,err);np.testing.assert_allclose(np.quantile(bd,[.025,.975]),q['conditional_IDcluster_bootstrap_CI95'],atol=1e-12,rtol=0);assert q['resampling_units']==400 and q['measurements_per_unit']==3;assert np.bincount(((A-B)>0).sum(0),minlength=4).tolist()==q['positive_partitions_per_ID_histogram0to3']
assert abs(r['individual_partitions']['original']['scores']['true']['RMSE']-12.761504566432446)<1e-12;c.check();rec={'passed':True,'all_partition_scalar_metrics_folds_top5_verified':True,'bootstrap3000_independent_index_replay_max_error':maxboot,'ID_cluster_400_units_shared_across_3_partitions_verified':True,'no1200row_independence_assumption':True,'original194_true_RMSE_exact':True,'report_SHA256':c.sha(R/'report.json'),'new_fits_or_inference':False};c.write(R/'score-validation.json',rec);c.emit(rec)
