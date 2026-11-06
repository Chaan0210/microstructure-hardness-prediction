"""Single affine correction from fully nested, train-only008 predictions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(_repo_path(ROOT, 'src')));assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import numpy as np,torch,joblib
from PIL import Image
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
import dinob_complement_064 as prior
from microstructure_v2.features import extract as extract_baseline
fv=prior.fv;h=prior.h;SEEDS=prior.SEEDS;OUT=_repo_path(ROOT, 'artifacts/nested-calibration-074');INNER_SEED=20270110

def check():
 prior.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for path,digest in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, path))==digest,path
 return p

def prepare():
 old=prior.check();pub=json.loads((_repo_path(ROOT, 'reports/candidate008-public-submission.json')).read_text());audit=json.loads((_repo_path(ROOT, 'artifacts/bias-audit-073/report.json')).read_text());assert all(v['ensemble008']['affine_slope_diagnostic_only']>1 for v in audit['results'].values());OUT.mkdir(exist_ok=False)
 p={'hypothesis':'008prediction-conditionedresiduals show underdispersion: diagnosticOLSslope1.1575/1.1612/1.1597,14of15foldslopes>1. A two-parameter affinecalibrator fittedonlyoninnerOOF ofoutertrain maycorrectensembleattenuation. Target-tailresiduals alonearenotevidence.',
 'single_primary':'For eachouterfold, GroupKFold3 shuffle20270110+fold onits320trainingrows. Refitentire008 insideeachinnertrain(213/214rows),predictinnerholdout. OLS y=a+b*p onpooled320innerOOFpairs. Applya+b*p008 to80outervalidationpredictions fromverified008. No clip,monotonicconstraint,binwise/splinefit,weightgrid,posthocshrink orlossswitch.',
 'entire_inner_model':'Unchanged008=.375baseline+.1875compactKRR+.1875TinyCNN+.125Smean3+.125Bmean3. Clonefixedbaseline3estimators;compactKRR selectexisting9grid usingitsown3foldtraining-onlyCV;CNN12880epochsD4last5weightaverage;S/Bsame3seeds100epochSmoothL1. Allscalers,normalizers,targetstatsfitinnertrainonly.',
 'leakage_control':'GlobalOOF008values areusedONLYasoutervalidationpredictionandunchangedreference,nevercalibrationtraining. Nestedcalibrationtrainingvaluesfromfreshmodelstrainedstrictlyinsideoutertrain. Kernelgrid hasonefurtherlevelofCV. No outervalidationlabelentersanyfit orchoice.',
 'seed_rule':'CNNbase20261003+outerfold+100*(innerfold+1). MLPbaseeachfixedseed+sameoffset. Existingfixedbaselineestimatorseedsretained. Kernelinnersplitseed20261005+offset.',
 'training_size_risk':'Calibrationpairsfrom213/214-trainedmodelsversusouter320;strongerattenuationwithlessdata couldovercorrect. Outerassessmenttestswholeprocedure;no adjustmentchosenfromouterperformance.',
 'screen':'OriginalpooledOOF strictlyimproves008 -> bothknownresplitsunchanged. Otherwise reject.',
 'confirmation':'All3pooledOOF strictlyimprove008 plusstrictleakage/reload/replayverification. Foldcounts/bootstrapdiagnosticsonly,notgates. Smallgainisexploratorynotindependentsignificance.',
 'resource':'ExistingCPU2/MPS;firstinnerfitissavedandusedasbenchmark,projected15fits<1200s,RSS<3GB,MPSdriver<3GB.15CNN+90MLPperpartition plusclassicalcomponents. No duplicatefirstfit. No newinstall/download/cost.',
 'inputs':{**old['inputs'],**{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/bias-audit-073/report.json','reports/dinob-complement-064-validation.json','artifacts/run-005/best_model.joblib','artifacts/frozen-validation-021/input-provenance.json','artifacts/run-012/report.json']}},
 'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['nested_calibration_074.py','validate_nested_calibration_074.py','frozen_validation.py','cnn_controls.py','compact_regression.py','dinob_mlp_head_063.py','dinob_mlp_063.py','dinob_complement_064.py','src/microstructure_v2/features.py','src/microstructure_v2/extra.py']},
 'protected':{**old['protected'],**pub['protected_sha256'],'reports/candidate008-public-submission.json':fv.sha(_repo_path(ROOT, 'reports/candidate008-public-submission.json'))},'scope':'Development400only;baselinefeaturesreextractedfrom400images so no mixed500cache isread. No holdout/testselection.','bias':fv.CAVEAT}
 fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'hypothesis':p['hypothesis'],'single_primary':p['single_primary']}),flush=True)

def extract_one(ident):
 with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:
  assert im.mode=='L' and im.size==(256,256);X,n=extract_baseline(np.array(im));image=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)/255
 return X,n,image,fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))

def extract():
 check();ids,*_=prior.old.inputs('original');bundle=joblib.load(_repo_path(ROOT, 'artifacts/run-005/best_model.joblib'));rows=[];start=time.perf_counter()
 for a in range(0,400,40):
  rows.extend(joblib.Parallel(n_jobs=2)(joblib.delayed(extract_one)(ident) for ident in ids[a:a+40]));print(json.dumps({'stage':'development_features','completed':len(rows),'seconds':time.perf_counter()-start}),flush=True)
 assert all(row[1]==bundle['feature_names'] for row in rows);X=np.stack([r[0] for r in rows]);images=np.stack([r[2] for r in rows])[:,None];assert X.shape==(400,795) and np.isfinite(X).all();old=json.loads((_repo_path(ROOT, 'artifacts/frozen-validation-021/input-provenance.json')).read_text());assert fv.array_sha(X)==old['baseline_selected_development_X_sha256']
 hashes={i:r[3] for i,r in zip(ids,rows)};assert hashes==old['development_image_sha256']
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),Xbase=X,images=images)
 report={'shape':list(X.shape),'feature_sha256':fv.sha(_repo_path(OUT, 'development_features.npz')),'baseline_selected_X_sha256':fv.array_sha(X),'image_sha256':hashes,'baseline_feature_names':bundle['feature_names'],'seconds':time.perf_counter()-start};fv.write(_repo_path(OUT, 'extraction.json'),report);print(json.dumps({'extracted':400,'exact_prior_development_features':True,'seconds':report['seconds']}),flush=True)

def inputs(split):
 ids,S,B,y,g,fid,r003,r007,Smean=prior.old.inputs(split);v=json.loads((_repo_path(ROOT, 'reports/dinob-complement-064-validation.json')).read_text());p=prior.OUT/split/'development_predictions.npz';assert fv.sha(p)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['fold'],fid);np.testing.assert_array_equal(z['y'],y);ref=z['mixed50'].copy()
 cache=_repo_path(OUT, 'development_features.npz');assert fv.sha(cache)==json.loads((_repo_path(OUT, 'extraction.json')).read_text())['feature_sha256']
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids;Xbase=z['Xbase'].copy();images=z['images'].copy()
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;K=z['X'].copy()
 panel=json.loads((_repo_path(ROOT, 'artifacts/run-012/report.json')).read_text());bundle=joblib.load(_repo_path(ROOT, 'artifacts/run-005/best_model.joblib'));recipes={n:clone(m) for n,m in bundle['models'].items()}
 return {'ids':ids,'S':S,'B':B,'y':y,'g':g,'fid':fid,'reference008':ref,'Xbase':Xbase,'images':images,'K':K,'blocks':panel['blocks'],'recipes':recipes,'weights':bundle['weights']}

def fit_cell(d,tr,va,outer_va,k,j,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();ids=d['ids'];y=d['y'];offset=k+100*(j+1);assert not set(tr)&set(va) and not set(tr)&set(outer_va) and not set(va)&set(outer_va);assert not set(d['g'][tr])&set(d['g'][va]);pred={};logs={}
 fitted={n:clone(m).fit(d['Xbase'][tr],y[tr]) for n,m in d['recipes'].items()};pred['baseline']=sum(d['weights'][n]*m.predict(d['Xbase'][va]) for n,m in fitted.items());joblib.dump({'models':fitted,'weights':d['weights'],'train_ids':[ids[i] for i in tr]},_repo_io(path/'baseline.joblib'));del fitted
 pred['kernel'],model,logs['kernel']=fv.kernel_fold(d['K'],y,d['g'],d['blocks'],tr,va,offset);joblib.dump({'model':model,'train_ids':[ids[i] for i in tr]},_repo_io(path/'kernel.joblib'));del model
 pred['cnn'],logs['cnn']=fv.cnn_fold(d['images'],y,tr,va,ids,offset,'mps',path/'cnn.pt')
 for arm in ('S','B'):
  X=d[arm];sc,ym,ys,xt,yt,bt,edges=prior.old.prep(X[tr],y[tr]);logs[arm]={}
  for seed in SEEDS:
   m,rec=h.fit(xt,yt,bt,seed+offset,0.);file=path/f'{arm}-{seed}.pt';torch.save({'model':m.state_dict(),'input_dim':X.shape[1],'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'bin_edges_unused':torch.tensor(edges),'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va],'seed':seed+offset,'config':h.CONFIG,'effective_aux_weight':0.},_repo_io(file));pred[f'{arm}_{seed}']=h.predict(m,sc.transform(X[va]),ym,ys);rec['checkpoint_sha256']=fv.sha(file);logs[arm][str(seed)]=rec
  pred[arm+'mean']=np.mean([pred[f'{arm}_{s}'] for s in SEEDS],axis=0)
 pred['ensemble008']=.375*pred['baseline']+.1875*pred['kernel']+.1875*pred['cnn']+.125*pred['Smean']+.125*pred['Bmean']
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,**pred)
 hashes={p.name:fv.sha(p) for p in path.iterdir() if p.is_file()};rec={'outer_fold':k,'inner_fold':j,'offset':offset,'train_indices':tr.tolist(),'validation_indices':va.tolist(),'outer_validation_indices':outer_va.tolist(),'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va],'logs':logs,'files_sha256':hashes,'seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};fv.write(path/'report.json',rec);print(json.dumps({'completed_inner':j,'outer_fold':k,'seconds':rec['seconds'],'n_train':len(tr),'n_calibration_rows':len(va)}),flush=True);return pred['ensemble008'],rec

def run(split):
 check();torch.set_num_threads(2)
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 d=inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);y=d['y'];fid=d['fid'];ref=d['reference008'];new=np.full(400,np.nan);folds=[];start=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);fd=out/f'fold-{k}';fd.mkdir();splits=list(GroupKFold(3,shuffle=True,random_state=INNER_SEED+k).split(d['S'][tr],y[tr],d['g'][tr]));ip=np.full(len(tr),np.nan);cells=[]
  for j,(it,iv) in enumerate(splits):
   pp,rec=fit_cell(d,tr[it],tr[iv],va,k,j,fd/f'inner-{j}');ip[iv]=pp;cells.append(rec)
   if split=='original' and k==0 and j==0:
    bench={'first_inner_seconds':rec['seconds'],'projected15_seconds':rec['seconds']*15,'rss_peak_bytes':rec['rss_peak_bytes'],'mps_driver_bytes':rec['mps_driver_bytes'],'first_fit_reused':True};bench['resource_pass']=bool(bench['projected15_seconds']<1200 and bench['rss_peak_bytes']<3e9 and bench['mps_driver_bytes']<3e9);fv.write(_repo_path(OUT, 'benchmark.json'),bench);assert bench['resource_pass']
  assert np.isfinite(ip).all();center=float(ip.mean());targetcenter=float(y[tr].mean());pc=ip-center;slope=float(pc@(y[tr]-targetcenter)/(pc@pc));intercept=targetcenter-slope*center;assert np.isfinite([slope,intercept]).all();new[va]=targetcenter+slope*(ref[va]-center)
  with (fd/'calibration.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,inner_oof=ip,prediction_center=center,target_center=targetcenter,slope=slope,intercept=intercept,outer_prediction=new[va])
  frec={'outer_fold':k,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'slope':slope,'intercept':intercept,'prediction_center':center,'target_center':targetcenter,'calibration_sha256':fv.sha(fd/'calibration.npz'),'inner_report_sha256':{f'inner-{j}/report.json':fv.sha(fd/f'inner-{j}/report.json') for j in range(3)},'outer_rmse_uncalibrated':fv.rmse(y[va],ref[va]),'outer_rmse_calibrated':fv.rmse(y[va],new[va])};fv.write(fd/'report.json',frec);folds.append(frec);print(json.dumps({'split':split,'completed_outer_folds':k+1,'slope_trainonly':slope,'elapsed_seconds':time.perf_counter()-start}),flush=True)
 comp=prior.old.comparison(y,ref,new,fid)
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=y,fold=fid,reference008=ref,primary=new)
 report={'split':split,'scores':{n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in [('reference008',ref),('primary',new)]},'comparison':comp,'calibration_slopes':[r['slope'] for r in folds],'fold_report_sha256':{f'fold-{k}/report.json':fv.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds':time.perf_counter()-start,'candidate_created':False};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':comp['gain']>0,'comparison':comp})
 check();print(json.dumps(report),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','extract','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'extract':extract}.get(a.mode,lambda:run(a.mode))()
