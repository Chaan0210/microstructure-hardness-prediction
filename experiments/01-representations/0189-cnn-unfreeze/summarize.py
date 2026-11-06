import common as c
from common import R,P,np
from study import D,A
import math
c.check();d=c.data('original');root=R/'resnet50';r=c.read(root/'report.json');v=c.read(R/'validation.json');assert v['passed'];z=c.load(root/'OOF.npz');y=z['y'];a=z['control_layer4'];b=z['two_stage'];ref=z['best021'];fold=z['fold'];rmse=lambda yy,p:float(np.sqrt(np.mean((yy-p)**2)))
for arr,key in [(a,'score_layer4'),(b,'score_two_stage')]:assert abs(math.sqrt(math.fsum(float(t-p)**2 for t,p in zip(y,arr))/400)-r[key]['RMSE']['standalone'])<1e-12
unique,inv=np.unique(d['g'],return_inverse=True)
def compare(p,q):
 fg=[rmse(y[fold==k],p[fold==k])-rmse(y[fold==k],q[fold==k]) for k in range(5)];rng=np.random.default_rng(202710189);boot=[]
 for _ in range(3000):
  counts=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv];boot.append(float(np.sqrt(counts@((y-p)**2)/counts.sum())-np.sqrt(counts@((y-q)**2)/counts.sum())))
 return {'RMSE_layer4':rmse(y,p),'RMSE_layer3_layer4':rmse(y,q),'gain_one_minus_two_stage':rmse(y,p)-rmse(y,q),'fold_gains':fg,'positive_folds':sum(x>0 for x in fg),'conditional_group_bootstrap_CI95':np.quantile(boot,[.025,.975]).tolist()}
paired={'standalone':compare(a,b),'fixed10_blend':compare(.9*ref+.1*a,.9*ref+.1*b)}
proxies=c.load(A/'boundary-proxies.npz');np.testing.assert_array_equal(proxies['ids'],d['ids']);fine=proxies['values'][:,0];high=np.zeros(400,bool)
for k in range(5):high[fold==k]=fine[fold==k]>=np.quantile(fine[fold!=k],.75)
prior=c.load(D/'diagnostic-masks.npz');np.testing.assert_array_equal(prior['ids'],d['ids']);large=prior['persistent_large33'].astype(bool);sparse=prior['sparse'].astype(bool);assert large.sum()==33
masks={'all400':np.ones(400,bool),'high_finecontrast_proxy':high,'other_finecontrast':~high,'posthoc_failure33':large,'outside_failure33':~large,'sparse_label_region':sparse,'dense_label_region':~sparse};groups={}
for name,mask in masks.items():
 groups[name]={'n':int(mask.sum()),'diagnostic_only':name!='all400'}
 for arm,pp in [('best021',ref),('layer4',a),('two_stage',b),('blend_layer4',.9*ref+.1*a),('blend_two_stage',.9*ref+.1*b)]:
  err=y[mask]-pp[mask];groups[name][arm]={'RMSE':float(np.sqrt(np.mean(err**2))),'MAE':float(np.mean(abs(err))),'bias_y_minus_prediction':float(err.mean())}
resources={}
for path in R.glob('*-resources.json'):
 q=c.read(path);t=q['trace'];resources[path.stem]={'seconds':t[-1]['seconds'],'max_RSS_plus_MPS':max(x['RSS_plus_MPS'] for x in t),'swap_growth':max(x['swap'] for x in t)-q['initial_swap']};assert resources[path.stem]['max_RSS_plus_MPS']<6e9 and resources[path.stem]['swap_growth']==0
training={}
for arm,pathroot in [('layer4',D/'resnet50'),('two_stage',root)]:
 fits=[c.read(pathroot/f'fold-{k}'/('outer-'+c.plan['locked_outer_recipes'][str(k)]['optimizer'])/'report.json') if arm=='layer4' else c.read(pathroot/f'fold-{k}/report.json') for k in range(5)]
 training[arm]={'train_RMSE_by_epoch':{str(ep):float(np.sqrt(np.mean([next(q['train_RMSE'] for q in fit['curve'] if q['epoch']==ep)**2 for fit in fits]))) for ep in [0,3,6,9,12]},'clip_fraction':float(np.mean([q['gradient_clip_fraction'] for q in fits])),'updates':[q['updates'] for q in fits]}
stability=[]
for k in range(5):
 q=c.read(root/f'fold-{k}/stability.json');assert q['passed'];g=q['gradient_trace'];f=q['feature_trace'];stability.append({'fold':k,'passed':True,'both_stages_changed':q['both_stages_changed'],'max_gradient_ratio':q['max_gradient_ratio'],'max_feature_RMS_ratio':q['max_feature_RMS_ratio'],'max_feature_maxabs_ratio':q['max_feature_maxabs_ratio'],'first_gradient':q['first_train_gradient_reference'],'max_preclip_total_gradient':max(t['total'] for t in g),'layer3_preclip_gradient_minmax':[min(t['layer3'] for t in g),max(t['layer3'] for t in g)],'layer4_preclip_gradient_minmax':[min(t['layer4'] for t in g),max(t['layer4'] for t in g)],'Z_RMS_minmax':[min(t['Z_RMS'] for t in f),max(t['Z_RMS'] for t in f)],'Z_maxabs_max':max(t['Z_maxabs'] for t in f),'feature_calls_checked':len(f),'updates_checked':len(g),'full_trace_SHA256':c.sha(root/f'fold-{k}/stability.json')})
score=r['score_two_stage'];close=score['RMSE']['blend10']>=score['RMSE']['best021'];decision='close_ResNet_local_adaptation_branch_keep_best021' if close else ('eligible_requires_parent_review_no_resplit_yet' if score['advance_gate'] else 'tiny_average_gain_below_promotion_gate_parent_review')
summary={'study':189,'decision':decision,'ResNet_local_adaptation_branch_closed':bool(close),'only_unfreeze_axis_changed':True,'input_fixed224':True,'paired_layer4_vs_two_stage':paired,'score_two_stage_vs021':score,'score_layer4_vs021':r['score_layer4'],'paired_protocol_proof':r['pairing_proof'],'hypothesis':c.plan['hypothesis'],'training':training,'stability':stability,'groups':groups,'validation':v,'selection_limit':c.plan['selection_limit'],'interpretation_limits':['Lowtrain/highOOFgapmotivatedafalsifiablecomplementaritytest,notanassumptionthatmoretrainablelayershelp.','Sameupdates/epochs/trials; trainableparametercountandFLOPsnecessarilyincrease. Actualcomputeisreported.','All5recipespreviouslychosenonlyinsideeachoutertrainin187;no189LRretuning.','Reused400andposthocResNetchoice;bootstrapconditionalnotselectionadjusted.','Finecontrastproxyisnottruegrainboundary;failure33posthocandevaluationonly.','Passing100xemergencystabilityguardmeansnodefinedexplosion,notproofdistributionstationarityorconvergence.'],'resources':resources,'measured_stage_seconds_sum':sum(x['seconds'] for x in resources.values()),'max_RSS_plus_MPS':max(x['max_RSS_plus_MPS'] for x in resources.values()),'main_outer_models':5,'additional_identical_label_poison_refits':1,'full_OOF_ID_order_finite_independent_fsum_verified':True,'protected_file_count':len(c.read(R/'preservation.json')),'protected_files_verified':True,'best021_SHA256':c.sha(P/'artifacts/submission-candidate-021.csv'),'new_install_download_cost_holdout_test_full500_resplit_CSV_upload':False,'cost':0,'source_SHA256':{n:c.sha(R/n) for n in ['plan.json','common.py','features.py','engine.py','stability.py','study.py','validate.py','summarize.py','launch.py']}}
c.check();c.write(R/'summary.json',summary);c.write(P/'reports/cnn-unfreeze-189-summary.json',{'summary_path':str((R/'summary.json').relative_to(P)),'summary_SHA256':c.sha(R/'summary.json'),'decision':decision,'paired':paired,'score_two_stage_vs021':score,'best021_preserved':True,'newCSV_upload':False});c.emit({'decision':decision,'paired':paired,'score_two_stage_vs021':score,'summary_SHA256':c.sha(R/'summary.json'),'max_RSS_plus_MPS':summary['max_RSS_plus_MPS'],'measured_stage_seconds':summary['measured_stage_seconds_sum']})
