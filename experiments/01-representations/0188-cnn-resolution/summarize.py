import common as c
from common import R,P,np
from study import D
import math
c.check();d=c.data('original');root=R/'resnet50';r=c.read(root/'report.json');v=c.read(R/'validation.json');assert v['passed'];z=c.load(root/'OOF.npz');y=z['y'];a=z['control224'];b=z['native256'];ref=z['best021'];fold=z['fold'];rmse=lambda yy,p:float(np.sqrt(np.mean((yy-p)**2)))
for arr,key in [(a,'scores224'),(b,'scores256')]:assert abs(math.sqrt(math.fsum(float(t-p)**2 for t,p in zip(y,arr))/400)-r[key]['RMSE']['standalone'])<1e-12
unique,inv=np.unique(d['g'],return_inverse=True)
def compare(p,q):
 fg=[rmse(y[fold==k],p[fold==k])-rmse(y[fold==k],q[fold==k]) for k in range(5)];rng=np.random.default_rng(202710188);boot=[]
 for _ in range(3000):
  counts=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv];boot.append(float(np.sqrt(counts@((y-p)**2)/counts.sum())-np.sqrt(counts@((y-q)**2)/counts.sum())))
 return {'RMSE224':rmse(y,p),'RMSE256':rmse(y,q),'gain224_minus256':rmse(y,p)-rmse(y,q),'fold_gains':fg,'positive_folds':sum(x>0 for x in fg),'conditional_group_bootstrap_CI95':np.quantile(boot,[.025,.975]).tolist()}
paired={'standalone':compare(a,b),'fixed10_blend':compare(.9*ref+.1*a,.9*ref+.1*b)}
proxies=c.load(R/'boundary-proxies.npz');np.testing.assert_array_equal(proxies['ids'],d['ids']);fine=proxies['values'][:,0];high=np.zeros(400,bool);thresholds=[]
for k in range(5):
 T=fold!=k;V=fold==k;q=float(np.quantile(fine[T],.75));high[V]=fine[V]>=q;thresholds.append({'fold':k,'train_q75':q,'outer_high_n':int(high[V].sum())})
prior=c.load(D/'diagnostic-masks.npz');np.testing.assert_array_equal(prior['ids'],d['ids']);large=prior['persistent_large33'].astype(bool);sparse=prior['sparse'].astype(bool);assert large.sum()==33
masks={'all400':np.ones(400,bool),'high_finecontrast_proxy':high,'other_finecontrast':~high,'posthoc_failure33':large,'outside_failure33':~large,'sparse_label_region':sparse,'dense_label_region':~sparse};group={}
for name,mask in masks.items():
 group[name]={'n':int(mask.sum()),'diagnostic_only':name!='all400','proxy_mean':dict(zip(proxies['columns'].tolist(),proxies['values'][mask].mean(0).tolist()))}
 for arm,pp in [('best021',ref),('CNN224',a),('CNN256',b),('blend224',.9*ref+.1*a),('blend256',.9*ref+.1*b)]:
  err=y[mask]-pp[mask];group[name][arm]={'RMSE':float(np.sqrt(np.mean(err**2))),'MAE':float(np.mean(abs(err))),'bias_y_minus_prediction':float(err.mean())}
resources={}
for path in R.glob('*-resources.json'):
 q=c.read(path);t=q['trace'];resources[path.stem]={'seconds':t[-1]['seconds'],'max_RSS_plus_MPS':max(x['RSS_plus_MPS'] for x in t),'swap_growth':max(x['swap'] for x in t)-q['initial_swap']};assert resources[path.stem]['max_RSS_plus_MPS']<6e9 and resources[path.stem]['swap_growth']==0
curves={}
for arm,pathroot in [('224',D/'resnet50'),('256',root)]:
 fits=[c.read(pathroot/f'fold-{k}'/('outer-'+c.plan['locked_outer_recipes'][str(k)]['optimizer'])/'report.json') if arm=='224' else c.read(pathroot/f'fold-{k}/report.json') for k in range(5)]
 curves[arm]={'train_RMSE_by_epoch':{str(ep):float(np.sqrt(np.mean([next(q['train_RMSE'] for q in fit['curve'] if q['epoch']==ep)**2 for fit in fits]))) for ep in [0,3,6,9,12]},'clip_fraction':float(np.mean([q['gradient_clip_fraction'] for q in fits])),'updates':[q['updates'] for q in fits]}
p=paired['standalone'];consistent=p['gain224_minus256']>0 and p['positive_folds']>=3 and p['conditional_group_bootstrap_CI95'][0]>0 and paired['fixed10_blend']['gain224_minus256']>0
summary={'study':188,'decision':'resolution_signal_requires_parent_review_no_expansion_yet' if consistent else 'no_consistent_resolution_signal_no_sweep_keep_best021','input_axis_only':True,'resolution_signal_gate':{'passed':bool(consistent),'criteria':c.read(R/'analysis-lock.json')['resolution_signal_gate']},'promote_gate_vs021':r['scores256']['advance_gate'],'paired224_vs256':paired,'score256_vs021':r['scores256'],'score224_vs021':r['scores224'],'groups':group,'fine_proxy_thresholds':thresholds,'boundary_proxy':c.read(R/'boundary-proxy-definition.json'),'training':curves,'selection_conditioning':c.plan['selection_limit'],'interpretation_limits':['Onlyonebackboneandtwoinputpolicies;noarchitectureceilingorconvergenceclaim.','Boundaryproxyincludesnoise/contrastandunknownphysicalpixelscale;224->256roundtripdiagnosticincludesextraresamplingnotusedbythe224model.','Failure33isposthocfrompreviousresidualsandnotanindependenttestorselectioncriterion.','No256innerretuning;all5outerrecipeswerepreviouslyselectedusingonlytheiroutertrainat224.','Wholedevelopment400hasadaptivehistory;bootstrapconditionalnotselectionadjusted.'],'validation':v,'resources':resources,'measured_stage_seconds_sum':sum(x['seconds'] for x in resources.values()),'max_RSS_plus_MPS':max(x['max_RSS_plus_MPS'] for x in resources.values()),'main_outer_models':5,'additional_identical_label_poison_refits':1,'full_OOF_ID_order_finite_independent_fsum_verified':True,'control224_reused_by_hash':c.plan['control224'],'protected_file_count':len(c.read(R/'preservation.json')),'protected_files_verified':True,'best021_SHA256':c.sha(P/'artifacts/submission-candidate-021.csv'),'new_install_download_cost_holdout_test_full500_resplit_CSV_upload':False,'cost':0,'no_other_resolution_sweep':True,'source_SHA256':{n:c.sha(R/n) for n in ['plan.json','common.py','features.py','study.py','validate.py','summarize.py','launch.py','analysis-lock.json']}}
c.check();c.write(R/'summary.json',summary);c.write(P/'reports/cnn-resolution-188-summary.json',{'summary_path':str((R/'summary.json').relative_to(P)),'summary_SHA256':c.sha(R/'summary.json'),'decision':summary['decision'],'paired':paired,'best021_preserved':True,'newCSV_upload':False});c.emit({'decision':summary['decision'],'paired':paired,'score256_vs021':r['scores256'],'summary_SHA256':c.sha(R/'summary.json'),'max_RSS_plus_MPS':summary['max_RSS_plus_MPS'],'measured_stage_seconds':summary['measured_stage_seconds_sum']})
