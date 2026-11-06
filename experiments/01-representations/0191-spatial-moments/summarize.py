import common as c
from common import R,P,np
import pipeline as e
import math
c.check();v=c.read(R/'validation.json');assert v['passed'];d=c.data('original');y=d['y'];fold=d['fold'];ref=d['blend_L_MLP'];unique,inv=np.unique(d['g'],return_inverse=True);rmse=lambda y,p:float(np.sqrt(np.mean((y-p)**2)))
def paired(a,b):
 fg=[rmse(y[fold==k],a[fold==k])-rmse(y[fold==k],b[fold==k]) for k in range(5)];rng=np.random.default_rng(202710191);boot=[]
 for _ in range(3000):
  count=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv];boot.append(float(np.sqrt(count@((y-a)**2)/count.sum())-np.sqrt(count@((y-b)**2)/count.sum())))
 return {'RMSE_mean':rmse(y,a),'RMSE_mean_std':rmse(y,b),'gain_mean_minus_mean_std':rmse(y,a)-rmse(y,b),'fold_gains':fg,'positive_folds':sum(t>0 for t in fg),'conditional_group_bootstrap_CI95':np.quantile(boot,[.025,.975]).tolist()}
results={};eligible=[]
for name in c.plan['models']:
 root=R/name;o=c.load(root/'OOF.npz');r=c.read(root/'report.json');assert c.sha(root/'OOF.npz')==r['OOF_SHA256'];scores=r['scores']
 for arm in c.plan['arms']:
  assert abs(math.sqrt(math.fsum(float(t-p)**2 for t,p in zip(y,o[arm]))/400)-scores[arm]['RMSE']['standalone'])<1e-12
 single=paired(o['mean'],o['mean_std']);blend=paired(.9*ref+.1*o['mean'],.9*ref+.1*o['mean_std']);gate=scores['mean_std']['advance_gate'] and single['gain_mean_minus_mean_std']>0 and blend['gain_mean_minus_mean_std']>0 and blend['positive_folds']>=3
 if gate:eligible.append(name)
 results[name]={'scores':scores,'paired_standalone':single,'paired_fixed10_blend':blend,'mean_std_advancement_gate':bool(gate),'fold_records':r['fold_records'],'extraction':c.read(root/'extraction.json'),'OOF_SHA256':r['OOF_SHA256']}
resources={}
for path in R.glob('*-resources.json'):
 q=c.read(path);tr=q['trace'];resources[path.stem]={'seconds':tr[-1]['seconds'],'max_RSS_plus_MPS':max(z['RSS_plus_MPS'] for z in tr),'swap_growth':max(z['swap'] for z in tr)-q['initial_swap']};assert resources[path.stem]['max_RSS_plus_MPS']<6e9 and resources[path.stem]['swap_growth']==0
s={'study':191,'decision':'report_passing_candidates_to_parent_before_resplit' if eligible else 'no_eligible_mean_std_candidate_keep_best021','eligible_mean_std_models':eligible,'results':results,'validation':v,'resources':resources,'measured_stage_seconds_sum':sum(z['seconds'] for z in resources.values()),'max_RSS_plus_MPS':max(z['max_RSS_plus_MPS'] for z in resources.values()),'protocol':{k:c.plan[k] for k in ['descriptor_policy','preprocess','head','evaluation']},'actual_fits':{'main100_epoch_heads':20,'extra_poison_replay_heads':4,'PCA_components':128,'head_parameters':8321,'epochs':100,'updates_per_head':500},'limits':['Existing scratch TinyCNN already uses mean/std; novelty is this frozen pretrained descriptor and matched PCA/head control.','Mean descriptor preserved exactly from188/190; each arm has a different train-only PCA subspace. The experiment estimates the full pooling-plus-compression pipeline, not an isolated std effect under fixed PCA axes.','PCA128 can discard signal. Failed comparison does not prove all spatial variability is useless or backbone capacity is exhausted.','Spatial std is a marginal dispersion statistic, not spatial arrangement or physical grain segmentation.','Reused development400 and conditional bootstrap are not independent validation or selection-adjusted evidence.'],'protected_file_count':len(c.read(R/'preservation.json')),'protected_files_verified':True,'best021_SHA256':c.sha(P/'artifacts/submission-candidate-021.csv'),'no_new_packages_downloads_cost':True,'holdout_test_access':False,'resplit_full500_newCSV_upload':False,'cost':0,'source_SHA256':{n:c.sha(R/n) for n in ['plan.json','common.py','launch.py','features.py','pipeline.py','study.py','validate.py','summarize.py']}}
c.check();c.write(R/'summary.json',s);c.write(P/'reports/spatial-moments-191-summary.json',{'summary_path':str((R/'summary.json').relative_to(P)),'summary_SHA256':c.sha(R/'summary.json'),'decision':s['decision'],'eligible_mean_std_models':eligible,'scores':{n:results[n]['scores'] for n in results},'best021_preserved':True,'no_holdout_test_full500_CSV_upload':True});c.emit({'decision':s['decision'],'eligible_mean_std_models':eligible,'summary_SHA256':c.sha(R/'summary.json'),'max_RSS_plus_MPS':s['max_RSS_plus_MPS'],'measured_stage_seconds':s['measured_stage_seconds_sum']})
