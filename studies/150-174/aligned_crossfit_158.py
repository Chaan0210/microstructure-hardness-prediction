"""Cross-fitted pooled residuals with the same anchor ensemble at deployment."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap_selector_144 as s
import validate_base_learning_curve_v2_127 as vk
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
x=s.x;a=s.a;e=s.e;n=s.n;OUT=_repo_path(ROOT, 'artifacts/aligned-crossfit-158')

def data(split):
 d=s.data(split);extra=e.cal.inputs(split);assert d['ids']==extra['ids'];np.testing.assert_array_equal(d['y'],extra['y']);np.testing.assert_array_equal(d['fid'],extra['fid'])
 for key in ['S','B','K','images']:d[key]=extra[key]
 return d

def head(obj,X):
 model=e.cal.h.Head(X.shape[1]);model.load_state_dict(obj['model']);return e.cal.h.predict(model,(X.astype(float)-obj['scaler_mean'].numpy())/obj['scaler_scale'].numpy(),obj['target_mean'],obj['target_std'])
def short_predict(obj,X,device='mps'):
 model=e.Small(obj['mean'].numpy(),obj['scale'].numpy(),obj['seed']);model.load_state_dict(obj['state']);model=model.to(device);pred=e.small_predict(model,X,np.arange(len(X)),device)*obj['target_std']+obj['target_mean'];del model
 if device=='mps':torch.mps.empty_cache()
 return pred

def prepare():
 s.check();OUT.mkdir(exist_ok=False);names=['aligned_crossfit_158.py','bootstrap_selector_144.py','nonlinear_residual_115.py','validate_nonlinear_residual_115.py','reports/residual-crossfit-118-scope-audit.json','reports/pooled-residual-119-summary.json','pooled_residual_119.py','reports/nested-selector-146-audit.json','reports/disagreement-156-audit.json','reports/candidate020-summary.json']
 for split in e.SPLITS:names += [f'artifacts/bootstrap-selector-144/{split}/report.json',f'artifacts/bootstrap-selector-144/validation-{split}.json']
 p={'study':'158 anchor-aligned cross-fitted stack','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prior119_correction':'119 already pooled320 own-label-excluded residuals with fixedrbf1 and NO innerselection. It was a valid outer comparison, not a demonstrated outer-label leak. Repeating fullTbase+fixedpooledresidual would duplicate it.','distinct_hypothesis':'119 retained target anchors trained213/214 but deploymentbase trained320.158 uses the SAME three A-only calibrated+shortB base models at outer inference, averaging their predictions, then adds one pooled320 residualfit. This aligns individual upstream train size and model construction between residual generation and deployment, while increasing each residualfit from107 to320.','definitions':'WithinouterT320/V80, A_j=TminusB_j. f_Aj=.9affine_Aj(.3KRR+.3CNNmean3identity+.2Smean3+.2Bmean3)+.1shortB_Aj; affine_Aj learned from A-only deepOOF. r_i=y_i-f_Aowner(i)(i) fori inB_j. b(V)=mean_j f_Aj(V). Matchedblock=b(V)+mean_j g_Bj(V); PRIMARY=b(V)+g_T(V). Unchanged018 is operatingreference. No new outer affine on pooled fitted residuals.','fixed_residual':'Existing n.fit kindrbf1 exactly,physics57+spatial528=585. Train-only MinMaxclip/StandardScaler/blockweights,medianpairdistance gamma,traincenter/trace normalization,residualRMS,alpha=1*n,effective df<=1. One fixed family/strength and no inner selection, bootstrap candidate weights,lambda sweep,none fallback or posthoc mixture.','matched_control':'Reuse45 current018 fullB rbf1 model objects with the same crossfit targets/features. Compute new aligned base for BOTH arms; only residual pooling differs between primary and matchedcontrol. No repeat of119 fullTbase+pooled arm and no use of020 extra feature.','leakage':'EveryT row has exactly one crossfitted target; its completeA upstream/affine/shortB excludes itsB andV. Allg_T statistics fitT only; no innerselector consuming contaminated pooled residuals. V excludes every upstream and downstream fit. Existing deeperA OOF calib scopes independently checked.','cache_reuse':{'existing_45_anchor_targets_and_deepOOF_affines':True,'existing_A_KRR_CNN3_outerV_predictions':'156 audited feature caches,45bundles; no newfits.','existing_A_S_B_and_shortB_checkpoints':'074/112,270S/Bheads+45shortB. Missing outerV predictions recomputed and checked on originalB queries.','existing45_fullB_rbf1':'144,exacttarget/feature/modelhashes verified.','new_models':'15 pooled320 RBF1 fits only; additional3poison replay fits. No CNN/MLP/backbone training.'},'strict_selected_alternative_not_run':{'extra_depth_base_bundles94_96':405,'current3seedCNN_fits_if_rebuilt':1215,'S_B_MLP_fits':2430,'shortB142_143_fits':135,'reason':'118 identified extra calibration depth required for clean innerselection. These are model-count estimates,not benchmarkedtime; current existing142 rawOOF caches alone cannot provide calibrated142 nuisances. Fixedrbf1 avoids this explosion.'},'criterion':'Run all3 existing partitions. Primary pooled untrimmedMSE must beat BOTH matchedblock and unchanged018, plusindependentverification. No fold/CI gate; controls not fallbackcandidates. Negative closes this aligned fixedstack with no alternate lambda or baseweights.','uncertainty':'10000 paired ID-group bootstrap seed20271218; all3partitions retained for each ID/group. Historical400adaptive reuse,anchor-overlap dependence and fixedfit CI limitations disclosed.','cautions':'Current3small residuals already jointly use320labels;107-to320 is perlearner,not3xindependentinformation. AveragingA models reduces queryvariance versusoneOOFanchor; alignment is not full distribution equality. Pooledpreprocessing/kernelstatistics alsochange. RBF1 may underfit; failure does not refute all properlynested stacks.','resources':'ExistingCPU2/MPS, zero packages/cost. Firstwholefoldbenchmark projected15<600sec,RSS<3GB. No new neuralfits. Test/holdout unavailable for selection; positivecandidate full500 would use3A333/334 basebundles plusone500residual.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':{str(p.relative_to(ROOT)):e.sha(p) for p in [_repo_path(ROOT, 'artifacts/submission-candidate-018.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-019.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-020.csv'),_repo_path(ROOT, 'deliverables/reproduction-018-v155.zip'),_repo_path(ROOT, 'deliverables/reproduction-011-014-015-v136.zip')]},'submission':'None; parentreported5/5 exhausted.018best and020secondary retained.','cost':0};e.write(_repo_path(OUT, 'plan.json'),p);e.write(_repo_path(ROOT, 'reports/aligned-crossfit-158-design.json'),p);e.emit({'locked158':True,'new_upstream_training':0,'new_pooled_models':15,'strict_nested_large_alternative_not_run':True,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 s.check();p=e.read(_repo_path(OUT, 'plan.json'))
 for f,h in {**p['hashes'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 return p

def anchor_node(d,split,k,j,path,create):
 A,B,M,short,_=d['cells'][k,j];V=np.flatnonzero(d['fid']==k);feature=_repo_path(ROOT, f'artifacts/disagreement-156/features/{split}/fold-{k}-block-{j}');fr=e.read(feature/'report.json');z=a.readnp(feature/'features.npz',fr['feature_sha256']);np.testing.assert_array_equal(z['anchor'],A);np.testing.assert_array_equal(z['block'],B);np.testing.assert_array_equal(z['outer'],V);source=e.cal.OUT/split/f'fold-{k}/inner-{j}';sr=e.read(source/'report.json');shortroot=e.OUT/split/f'fold-{k}/inner-{j}/shortB';rr=e.read(shortroot/'report.json');parent=e.read(shortroot.parent/'report.json');assert e.sha(shortroot/'report.json')==parent['model_reports']['shortB'];refs=list(fr['models'])
 for ref in refs:assert e.sha(_repo_path(ROOT, ref['path']))==ref['sha256']
 if create:
  path.mkdir(exist_ok=False);PV=np.zeros((len(V),5));PV[:,1]=z['kernel_V'];PV[:,2]=z['cnn_V'].mean(0);b_error=0.;functional_error=0.
  for arm,col in [('S',3),('B',4)]:
   sc=StandardScaler().fit(d[arm][A]);bp=[];vp=[]
   for seed in e.cal.SEEDS:
    file=source/f'{arm}-{seed}.pt';assert e.sha(file)==sr['files_sha256'][file.name];obj=torch.load(_repo_io(file),map_location='cpu',weights_only=True);assert obj['train_ids']==[d['ids'][i] for i in A];assert obj['target_mean']==float(d['y'][A].mean()) and obj['target_std']==float(d['y'][A].std());np.testing.assert_array_equal(obj['scaler_mean'].numpy(),sc.mean_);np.testing.assert_array_equal(obj['scaler_scale'].numpy(),sc.scale_);pb=head(obj,d[arm][B]);pv=head(obj,d[arm][V]);bp.append(pb);vp.append(pv);err=float(abs(vk.manual_head(obj,d[arm][V])-pv).max());assert err<1e-4;functional_error=max(functional_error,err);refs.append({'path':str(file.relative_to(ROOT)),'sha256':e.sha(file)})
   err=float(abs(np.mean(bp,0)-M[:,col]).max());assert err<1e-4;b_error=max(b_error,err);PV[:,col]=np.mean(vp,0)
  file=shortroot/'model.pt';assert e.sha(file)==rr['model_sha256'];obj=torch.load(_repo_io(file),map_location='cpu',weights_only=True);assert obj['train_ids']==[d['ids'][i] for i in A] and obj['train_indices']==A.tolist();assert obj['target_mean']==float(d['y'][A].mean()) and obj['target_std']==float(d['y'][A].std());sc=StandardScaler().fit(d['B'][A]);np.testing.assert_array_equal(obj['mean'].numpy(),sc.mean_);np.testing.assert_array_equal(obj['scale'].numpy(),sc.scale_);pb=short_predict(obj,d['B'][B]);sv=short_predict(obj,d['B'][V]);err=float(abs(pb-short).max());assert err<1e-4;b_error=max(b_error,err);refs.append({'path':str(file.relative_to(ROOT)),'sha256':e.sha(file)});cal,coef=a.affine(a.mixture(d['deepP'][k,j],x.ARM),d['y'][A],a.mixture(PV,x.ARM));base=.9*cal+.1*sv;ap,_=a.affine(a.mixture(d['deepP'][k,j],x.ARM),d['y'][A],a.mixture(M,x.ARM));anchor=.9*ap+.1*short;oldroot=s.q.OUT/split/f'adjacent-fold-{k}/block-{j}';old=e.read(oldroot/'report.json');oz=a.readnp(oldroot/'predictions.npz',old['predictions_sha256']);np.testing.assert_array_equal(anchor,oz['anchor_predictions']);assert not set(A)&set(B) and not(set(A)|set(B))&set(V) and not set(d['g'][A])&set(d['g'][B]);assert set(A)|set(B)==set(np.flatnonzero(d['fid']!=k))
  with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),anchor_indices=A,block_indices=B,outer_indices=V,components_V=PV,short_V=sv,base_V=base,anchor_B=anchor)
  rec={'models':refs,'calibrator':coef,'original_B_prediction_max_error':b_error,'independent_head_functional_max_error':functional_error,'predictions_sha256':e.sha(path/'predictions.npz'),'source_anchor_report_sha256':e.sha(oldroot/'report.json'),'same_A_model_scope_B_and_V':True};e.write(path/'report.json',rec)
 else:rec=e.read(path/'report.json')
 saved=a.readnp(path/'predictions.npz',rec['predictions_sha256']);np.testing.assert_array_equal(saved['anchor_indices'],A);np.testing.assert_array_equal(saved['block_indices'],B);np.testing.assert_array_equal(saved['outer_indices'],V)
 for ref in rec['models']:assert e.sha(_repo_path(ROOT, ref['path']))==ref['sha256']
 return saved,rec

def fold(d,split,k,path,create_features=True):
 path.mkdir(exist_ok=False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);lookup={int(i):ix for ix,i in enumerate(T)};targets=np.full(len(T),np.nan);seen=[];bases=[];block=[];refs={}
 for j in range(3):
  featurepath=_repo_path(OUT, split, f'anchor-{k}-{j}');z,r=anchor_node(d,split,k,j,featurepath,create_features);B=z['block_indices'];where=[lookup[int(i)] for i in B];targets[where]=z['anchor_B'];seen.extend(B.tolist());bases.append(z['base_V']);src=s.OUT/split/f'fold-{k}-block-{j}';sr=e.read(src/'report.json');file=src/'rbf1.joblib';assert e.sha(file)==sr['models_sha256'][file.name];obj=joblib.load(_repo_io(file));np.testing.assert_array_equal(obj['train_indices'],B);pp=n.predict(obj['model'],d['X'][V]);oz=a.readnp(src/'predictions.npz',sr['predictions_sha256']);np.testing.assert_array_equal(pp,oz['full_candidate_predictions'][3]);block.append(pp);refs[str(file.relative_to(ROOT))]=e.sha(file);refs[str((featurepath/'report.json').relative_to(ROOT))]=e.sha(featurepath/'report.json')
 assert sorted(seen)==T.tolist() and np.isfinite(targets).all() and not set(d['g'][T])&set(d['g'][V]);res=d['y'][T]-targets;m=n.fit(d['X'][T],res,d['blocks'],'rbf1');pooled=n.predict(m,d['X'][V]);joblib.dump({'model':m,'train_indices':T,'validation_indices':V},_repo_io(path/'pooled.joblib'));bases=np.stack(bases);block=np.stack(block);base=bases.mean(0);primary=base+pooled;matched=base+block.mean(0)
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train=T,outer=V,targets=targets,residual=res,anchor_bases=bases,base=base,block_corrections=block,pooled_correction=pooled,primary=primary,matched_block=matched)
 r={'source_hashes':refs,'model_sha256':e.sha(path/'pooled.joblib'),'predictions_sha256':e.sha(path/'predictions.npz'),'n_residual_train':len(T),'n_individual_block':[len(d['cells'][k,j][1]) for j in range(3)],'kind':'rbf1','alpha':m['alpha'],'effective_df':m['effective_df'],'no_inner_selection':True};e.write(path/'report.json',r);return primary,matched,r

def run(split):
 check();d=data(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);primary=np.full(400,np.nan);matched=primary.copy();hashes={};start=time.perf_counter()
 for k in range(5):
  dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);V=np.flatnonzero(d['fid']==k);primary[V],matched[V],r=fold(dd,split,k,out/f'fold-{k}');hashes[f'fold-{k}/report.json']=e.sha(out/f'fold-{k}/report.json')
  if split=='original' and k==0:
   b={'first_wholefold_seconds':time.perf_counter()-start,'projected15_seconds':15*(time.perf_counter()-start),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};b['passed']=b['projected15_seconds']<600 and b['RSS_bytes']<3e9;e.write(_repo_path(OUT, 'benchmark.json'),b);assert b['passed'];e.emit(b)
  e.emit({'split':split,'fold_complete':k,'seconds':time.perf_counter()-start})
 rr=e.read(s.OUT/split/'report.json');z=a.readnp(s.OUT/split/'predictions.npz',rr['predictions_sha256']);assert z['ids'].tolist()==d['ids']
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),groups=d['g'],y=d['y'],fold=d['fid'],reference018=z['primary'],matched_block=matched,primary=primary)
 r={'split':split,'RMSE':{'reference018':e.rmse(d['y'],z['primary']),'matched_block':e.rmse(d['y'],matched),'primary':e.rmse(d['y'],primary)},'fold_reports_sha256':hashes,'predictions_sha256':e.sha(out/'predictions.npz'),'new_pooled_models':5,'new_neural_fits':0,'seconds':time.perf_counter()-start};e.write(out/'report.json',r);check();e.emit(r['RMSE'])

def validate(split):
 check();d=data(split);out=_repo_path(OUT, split);rr=e.read(out/'report.json');saved=a.readnp(out/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);count=0;maxerr=0.;cpuerr=0.
 for k in range(5):
  dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);path=out/f'fold-{k}';r=e.read(path/'report.json');assert e.sha(path/'report.json')==rr['fold_reports_sha256'][f'fold-{k}/report.json'];z=a.readnp(path/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(z['train'],T);np.testing.assert_array_equal(z['outer'],V);lookup={int(i):ix for ix,i in enumerate(T)};target=np.full(len(T),np.nan);seen=[];bases=[]
  for j in range(3):
   az,ar=anchor_node(dd,split,k,j,out/f'anchor-{k}-{j}',False);A,B,M,short,_=dd['cells'][k,j];mb=.9*s.q.independent_affine(a.mixture(dd['deepP'][k,j],x.ARM),d['y'][A],a.mixture(az['components_V'],x.ARM))+.1*az['short_V'];np.testing.assert_allclose(mb,az['base_V'],atol=1e-9,rtol=0);ab=.9*s.q.independent_affine(a.mixture(dd['deepP'][k,j],x.ARM),d['y'][A],a.mixture(M,x.ARM))+.1*short;np.testing.assert_allclose(ab,az['anchor_B'],atol=1e-9,rtol=0);target[[lookup[int(i)] for i in B]]=az['anchor_B'];seen.extend(B.tolist());bases.append(az['base_V']);file=s.OUT/split/f'fold-{k}-block-{j}/rbf1.joblib';assert e.sha(file)==r['source_hashes'][str(file.relative_to(ROOT))];m=joblib.load(_repo_io(file))['model'];man=s.q.vn.manual(m,d['X'][B],d['y'][B]-az['anchor_B'],d['X'][V],d['blocks']);err=float(abs(man-z['block_corrections'][j]).max());assert err<2e-9;maxerr=max(maxerr,err);count+=1
   if k==j==0:
    ref=ar['models'][-1];obj=torch.load(_repo_path(ROOT, ref['path']),weights_only=True,map_location='cpu');pp=short_predict(obj,d['B'][V],'cpu');cpuerr=float(abs(pp-az['short_V']).max());assert cpuerr<1e-3
  assert sorted(seen)==T.tolist();np.testing.assert_array_equal(target,z['targets']);np.testing.assert_array_equal(d['y'][T]-target,z['residual']);np.testing.assert_array_equal(np.mean(bases,0),z['base']);assert e.sha(path/'pooled.joblib')==r['model_sha256'];obj=joblib.load(_repo_io(path/'pooled.joblib'));np.testing.assert_array_equal(obj['train_indices'],T);np.testing.assert_array_equal(obj['validation_indices'],V);m=obj['model'];assert m['kind']=='rbf1' and m['n_train']==320 and m['alpha']==320 and m['effective_df']<=1+1e-8;actual=n.predict(m,d['X'][V]);man=s.q.vn.manual(m,d['X'][T],z['residual'],d['X'][V],d['blocks']);err=float(abs(man-actual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,z['pooled_correction']);np.testing.assert_array_equal(z['base']+actual,saved['primary'][V]);np.testing.assert_array_equal(z['base']+z['block_corrections'].mean(0),saved['matched_block'][V]);count+=1
  if k==0:
   pert=dict(dd);pert['y']=d['y'].copy();pert['y'][V]+=100000;pp,mm,_=fold(pert,split,0,_repo_path(OUT, f'poison-{split}'),False);np.testing.assert_array_equal(pp,saved['primary'][V]);np.testing.assert_array_equal(mm,saved['matched_block'][V])
 r={'passed':True,'split':split,'residual_models_verified':count,'new_pooled_models_verified':5,'independent_max_error':maxerr,'CPU_MPS_shortB_max_error':cpuerr,'one_target_per_outertrain_ID':True,'own_label_and_outer_label_exclusion_checked':True,'no_inner_selector':True,'outer_target_poison_exact':True,'upstream_training_exclusion_certificate':e.sha(_repo_path(ROOT, 'reports/nested-selector-146-audit.json')),'report_sha256':e.sha(out/'report.json')};e.write(_repo_path(OUT, f'validation-{split}.json'),r);check();e.emit(r)

def summary():
 check();E={k:[] for k in ['reference018','matched_block','primary']};scores={};ids=None;groups=None
 for split in e.SPLITS:
  r=e.read(_repo_path(OUT, split, 'report.json'));v=e.read(_repo_path(OUT, f'validation-{split}.json'));assert v['passed'] and v['report_sha256']==e.sha(_repo_path(OUT, split, 'report.json'));z=a.readnp(_repo_path(OUT, split, 'predictions.npz'),r['predictions_sha256']);scores[split]=r['RMSE']
  if ids is None:ids=z['ids'];groups=z['groups']
  else:np.testing.assert_array_equal(ids,z['ids']);np.testing.assert_array_equal(groups,z['groups'])
  for key in E:E[key].append((z[key]-z['y'])**2)
 E={k:np.stack(v) for k,v in E.items()};M={k:float(v.mean()) for k,v in E.items()};rng=np.random.default_rng(20271218);gi=[np.flatnonzero(groups==g) for g in np.unique(groups)];gains={key:[] for key in ['reference018','matched_block']}
 for _ in range(10000):
  ii=np.concatenate([gi[i] for i in rng.integers(0,len(gi),len(gi))])
  for key in gains:gains[key].append(float(np.sqrt(E[key][:,ii].mean())-np.sqrt(E['primary'][:,ii].mean())))
 r={'study':'158 anchor-aligned crossfit pooled320','advance':M['primary']<min(M['reference018'],M['matched_block']),'validation_passed':True,'RMSE_by_partition':scores,'pooled_MSE':M,'pooled_RMSE':{k:float(np.sqrt(v)) for k,v in M.items()},'conditional_ID_group_gain_CI95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in gains.items()},'unique_IDs':len(ids),'unique_groups':len(gi),'same400_adaptive_history':True,'no_inner_selection':True,'new_upstream_models':0,'new_pooled_models':15,'current_best':'018','020_secondary_preserved':True,'test_or_holdout_read':False,'cost':0,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))};e.write(_repo_path(OUT, 'decision.json'),r);e.write(_repo_path(ROOT, 'reports/aligned-crossfit-158-summary.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','summary',*e.SPLITS,*['validate-'+v for v in e.SPLITS]]);mode=p.parse_args().mode
 if mode=='prepare':prepare()
 elif mode=='summary':summary()
 elif mode.startswith('validate-'):validate(mode[9:])
 else:run(mode)
