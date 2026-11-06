"""Fixed eight-view mean at every CNN prediction level of018, no neural retraining."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,csv,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import directional_energy_148 as core
import cnn_ensemble_inference as views
import numpy as np,torch
s=core.s;x=s.x;a=s.a;e=s.e;OUT=_repo_path(ROOT, 'artifacts/cnn-view-mean-151')

def outer_file(split):return _repo_path(ROOT, 'artifacts/run-020/development_oof.csv') if split=='original' else _repo_path(ROOT, f'artifacts/frozen-validation-021/seed-{split}/development_oof.csv')
def prepare():
 s.check();OUT.mkdir(exist_ok=False);assert not e.read(_repo_path(ROOT, 'artifacts/colocated-intensity-149/decision.json'))['advance'];names=['cnn_view_mean_151.py','cnn_ensemble_inference.py','directional_energy_148.py','cnn_seed_mean_133.py','reports/cnn-view-150-diagnostic.json','reports/directional-energy-148-summary.json','reports/colocated-intensity-149-summary.json','reports/nested-selector-146-audit.json','reports/candidate018-public-submission.json']
 for split in e.SPLITS:names += [str(outer_file(split).relative_to(ROOT)),f'artifacts/bootstrap-selector-144/{split}/report.json',f'artifacts/bootstrap-selector-144/{split}/predictions.npz',f'artifacts/cnn-seed-mean-133/validation-{split}.json']
 r={'study':'151 fixedCNN8viewgroupmean inside018','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evidence':e.read(_repo_path(ROOT, 'reports/cnn-view-150-diagnostic.json')),'recipe':'Exactly018 exceptCNNcomponent uses equalmean3seeds x8D4views; same3checkpoints,fullgroupuniformweight,noseed/viewsubset. D4CNNtrainingalreadyfixed. RefitouteraffinefromD4middleOOF,anchorA-affinefromD4deepOOF,thenentire585featurebootstrapresidualcandidateprocedure onnewhonestresiduals. DINO/shortB/physics/spatialfeatures unchanged;no148/149features.', 'duplicate_distinction':'D4itselfwasalreadytested020/021/004underolderbaseline/KRRensemble.151testsfullnestedcurrent018procedure. Reusesall45outerD4CNNpredictions; computesonlymissing180nestednodes*3models fromexisting074/112/133checkpoints. No newCNN/backbonefit andno claimnewaugmentationmethod.', 'nested':'Everymiddle/deepcache readexacttrain/validationindices andchecktrainIDs/scalers; sameA/B/V/C/Das146audit. No replacingonlyouterCNNwhilekeepingidentitycalibration/anchors. NewresidualinnerfitsBminusH excludeHlabels throughA-onlyanchor. NewbootstrapweightsfromBinnerOOFonly.', 'criterion':'All3existingpartitions,onefixedrecipe. AdvanceonlypooleduntrimmedMSE below018 andindependentvalidationpasses. Noearlyperformancegate,perfold/CIrequirement orposthocidentityD4weight. Ifnegativepreserve018 andstopthisdirection.', 'validation':'AllnewnestedcheckpointtrainIDs/normalization andidentityview match133cachedseedpredictions<=1e-4; each24viewmean serializedandrecomputed. CPU/MPS8viewchecks onmiddle0/deep0perpartition,token-independentD4 closure fixture. Exact018controlcache; independentOLS/900residualmodels/bootstrapweights; outerlabelpoisoncompleteonefoldperpartition. Existingupstreamtrainingpoison146certificate reused.', 'resources':'CPU2/MPSexistinguv. Firstnestednode projected180nodes<1800sec,RSS<3GB. No newpackages/cost. No holdout/testbeforepositivefinaldecision. Full500same24viewformulaatnested/all500levels,existingcheckpointsreused; testonlyafterlockandtrainingvalidation.', 'uncertainty':'10000paired400IDclusterbootstrapseed20271210; same400historicaladaptivebias,overlappingpartitions. D4trainedCNN doesnotproveallphysicalorientationinvariance; itsgroupmean isfixedinductiveassumption.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':{f'artifacts/submission-candidate-{c}.csv':e.sha(_repo_path(ROOT, f'artifacts/submission-candidate-{c}.csv')) for c in ['017','018']},'cost':0};e.write(_repo_path(OUT, 'plan.json'),r);e.emit({'locked151':True,'new_neural_fits':0,'missing_nested_prediction_nodes':180,'sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 r=e.read(_repo_path(OUT, 'plan.json'))
 for f,h in {**r['hashes'],**r['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 return r

def node(d,T,V,models,oldseeds,path,create):
 if create:
  path.mkdir(exist_ok=False);start=time.perf_counter();allviews=[];records=[];norm={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())}
  for si,file in enumerate(models):
   obj=torch.load(_repo_io(file),map_location='cpu',weights_only=True);assert obj['train_ids']==[d['ids'][i] for i in T] and obj['normalization']==norm;vv=views.predict_views(d['images'][V],file,'mps');err=float(abs(vv[0]-oldseeds[si]).max());assert err<1e-4;allviews.append(vv);records.append({'path':str(file.relative_to(ROOT)),'sha256':e.sha(file),'identity_max_error':err,'train_scope_and_normalization_exact':True})
  allviews=np.stack(allviews);mean=allviews.mean((0,1))
  with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train=T,validation=V,views=allviews,mean=mean)
  r={'train':T.tolist(),'validation':V.tolist(),'models':records,'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};e.write(path/'report.json',r)
 else:
  r=e.read(path/'report.json');assert r['train']==T.tolist() and r['validation']==V.tolist()
  for file,record in zip(models,r['models']):assert str(file.relative_to(ROOT))==record['path'] and e.sha(file)==record['sha256']
  z=a.readnp(path/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(z['train'],T);np.testing.assert_array_equal(z['validation'],V);np.testing.assert_allclose(z['views'][:,0],oldseeds,atol=1e-4,rtol=0);mean=z['views'].mean((0,1));np.testing.assert_array_equal(mean,z['mean'])
 return mean,r

def data(split):
 d=s.data(split);r=e.read(s.OUT/split/'report.json');z=a.readnp(s.OUT/split/'predictions.npz',r['predictions_sha256']);d['reference018']=z['primary'];return d

def replace(d,split,k,create):
 dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k)
 with outer_file(split).open() as f:rows={r['ID']:r for r in csv.DictReader(f)}
 assert set(rows)==set(d['ids']);outer=np.array([float(rows[i]['cnn_mean3_d4']) for i in d['ids']]);dd['P'][V,2]=outer[V];lookup={int(i):j for j,i in enumerate(T)};root=_repo_path(OUT, split, f'cnn-fold-{k}')
 if create:root.mkdir(exist_ok=False)
 for j in range(3):
  A,B,P,short,ref=dd['cells'][k,j];oldroot=x.OUT/split/f'cnn-fold-{k}/middle-{j}';oldrec=e.read(oldroot/'report.json');old=a.readnp(oldroot/'predictions.npz',oldrec['predictions_sha256']);models=[e.cal.OUT/split/f'fold-{k}/inner-{j}/cnn.pt',*[oldroot/f'seed-{seed}.pt' for seed in x.SEEDS[1:]]];pp,r=node(d,A,B,models,old['seeds'],root/f'middle-{j}',create);Q=P.copy();Q[:,2]=pp;dd['cells'][k,j]=(A,B,Q,short,ref);dd['middleP'][k][[lookup[int(i)] for i in B],2]=pp;la={int(i):ii for ii,i in enumerate(A)}
  if create and split=='original' and k==j==0:
   bench={'first_node_seconds':r['seconds'],'projected180_seconds':180*r['seconds'],'RSS_bytes':r['RSS_bytes']};bench['passed']=bench['projected180_seconds']<1800 and bench['RSS_bytes']<3e9;e.write(_repo_path(OUT, 'benchmark.json'),bench);assert bench['passed'];e.emit(bench)
  for ell in range(3):
   source=e.OUT/split/f'fold-{k}/inner-{j}/deep-{ell}';C,D,_=a.bundle(source);assert set(C)|set(D)==set(A) and not(set(C)|set(D))&(set(B)|set(V));oldroot=x.OUT/split/f'cnn-fold-{k}/deep-{j}-{ell}';oldrec=e.read(oldroot/'report.json');old=a.readnp(oldroot/'predictions.npz',oldrec['predictions_sha256']);models=[source/'cnn.pt',*[oldroot/f'seed-{seed}.pt' for seed in x.SEEDS[1:]]];pp,r=node(d,C,D,models,old['seeds'],root/f'deep-{j}-{ell}',create);dd['deepP'][k,j][[la[int(i)] for i in D],2]=pp
  e.emit({'split':split,'outer':k,'D4_nested_block_complete':j,'new_prediction_cache':create})
 return dd

def fold(d,split,k,path):
 path.mkdir(exist_ok=False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);cal,coef=a.affine(a.mixture(d['middleP'][k],x.ARM),d['y'][T],a.mixture(d['P'][V],x.ARM));base=.9*cal+.1*d['short'][V];cor=[];anchors=[];hashes={}
 for j in range(3):
  A,B,P,short,_=d['cells'][k,j];ap,ac=a.affine(a.mixture(d['deepP'][k,j],x.ARM),d['y'][A],a.mixture(P,x.ARM));anchor=.9*ap+.1*short;pp,rec=core.system(d,B,anchor,V,a.r.CVSEED+100*k+j,s.SEED+10000*e.SPLITS.index(split)+100*k+j,path/f'block-{j}');cor.append(pp);anchors.append(ac);hashes[f'block-{j}/report.json']=e.sha(path/f'block-{j}/report.json')
 pred=base+np.mean(cor,0)
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),outer_indices=V,base=base,corrections=np.stack(cor),primary=pred)
 r={'outer_calibrator':coef,'anchor_calibrators':anchors,'block_reports_sha256':hashes,'predictions_sha256':e.sha(path/'predictions.npz')};e.write(path/'report.json',r);return pred,r

def run(split):
 check();d=data(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full(400,np.nan);hashes={};start=time.perf_counter()
 for k in range(5):
  dd=replace(d,split,k,True);V=np.flatnonzero(d['fid']==k);pred[V],rec=fold(dd,split,k,out/f'fold-{k}');hashes[f'fold-{k}/report.json']=e.sha(out/f'fold-{k}/report.json');e.emit({'split':split,'fold_complete':k,'seconds':time.perf_counter()-start})
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference018=d['reference018'],primary=pred)
 r={'split':split,'RMSE':{'reference018':e.rmse(d['y'],d['reference018']),'primary':e.rmse(d['y'],pred)},'fold_reports_sha256':hashes,'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start,'new_neural_fits':0};e.write(out/'report.json',r);check();e.emit(r)

def validate(split):
 check();d=data(split);out=_repo_path(OUT, split);rr=e.read(out/'report.json');saved=a.readnp(out/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);count=0;maxerr=0.;cpuerr=0.
 for k in range(5):
  dd=replace(d,split,k,False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);root=out/f'fold-{k}';fr=e.read(root/'report.json');assert e.sha(root/'report.json')==rr['fold_reports_sha256'][f'fold-{k}/report.json'];fz=a.readnp(root/'predictions.npz',fr['predictions_sha256']);manual=.9*s.q.independent_affine(a.mixture(dd['middleP'][k],x.ARM),d['y'][T],a.mixture(dd['P'][V],x.ARM))+.1*dd['short'][V];np.testing.assert_allclose(manual,fz['base'],atol=1e-9,rtol=0);cor=[]
  for j in range(3):
   A,B,P,short,_=dd['cells'][k,j];ap,ac=a.affine(a.mixture(dd['deepP'][k,j],x.ARM),d['y'][A],a.mixture(P,x.ARM));assert ac==fr['anchor_calibrators'][j];anchor=.9*ap+.1*short;manual=.9*s.q.independent_affine(a.mixture(dd['deepP'][k,j],x.ARM),d['y'][A],a.mixture(P,x.ARM))+.1*short;np.testing.assert_allclose(manual,anchor,atol=1e-9,rtol=0);path=root/f'block-{j}';assert e.sha(path/'report.json')==fr['block_reports_sha256'][f'block-{j}/report.json'];pp,c,err=core.verify_system(dd,B,anchor,V,path);cor.append(pp);count+=c;maxerr=max(maxerr,err)
  np.testing.assert_array_equal(fz['base']+np.mean(cor,0),saved['primary'][V])
  if k==0:
   for nodepath in [out/'cnn-fold-0/middle-0',out/'cnn-fold-0/deep-0-0']:
    nr=e.read(nodepath/'report.json');z=a.readnp(nodepath/'predictions.npz',nr['predictions_sha256']);VV=z['validation']
    for si,model in enumerate(nr['models']):
     pred=views.predict_views(d['images'][VV],_repo_path(ROOT, model['path']),'cpu');err=float(abs(pred-z['views'][si]).max());assert err<1e-3;cpuerr=max(cpuerr,err)
   pert=dict(dd);pert['y']=dd['y'].copy();pert['y'][V]+=100000;pp,rec=fold(pert,split,0,_repo_path(OUT, f'poison-{split}'));np.testing.assert_array_equal(pp,saved['primary'][V]);assert rec['outer_calibrator']==fr['outer_calibrator'] and rec['anchor_calibrators']==fr['anchor_calibrators']
 r={'passed':True,'split':split,'models_verified':count,'independent_residual_max_error':maxerr,'CPU_MPS_nested_view_max_error':cpuerr,'nested_prediction_nodes_verified':60,'outer_target_poison_exact':True,'inner_anchor_exclusion_certificate':e.sha(_repo_path(ROOT, 'reports/nested-selector-146-audit.json')),'report_sha256':e.sha(out/'report.json')};e.write(_repo_path(OUT, f'validation-{split}.json'),r);check();e.emit(r)

def summary():
 check();E={k:[] for k in ['reference018','primary']};scores={}
 for split in e.SPLITS:
  r=e.read(_repo_path(OUT, split, 'report.json'));v=e.read(_repo_path(OUT, f'validation-{split}.json'));assert v['passed'] and v['report_sha256']==e.sha(_repo_path(OUT, split, 'report.json'));z=a.readnp(_repo_path(OUT, split, 'predictions.npz'),r['predictions_sha256']);scores[split]=r['RMSE']
  for k in E:E[k].append((z[k]-z['y'])**2)
 E={k:np.stack(v) for k,v in E.items()};M={k:float(v.mean()) for k,v in E.items()};rng=np.random.default_rng(20271210);g=[]
 for _ in range(10000):
  ii=rng.integers(0,400,400);g.append(float(np.sqrt(E['reference018'][:,ii].mean())-np.sqrt(E['primary'][:,ii].mean())))
 r={'study':'151 fixedCNN3seed8viewmean inside018','advance':M['primary']<M['reference018'],'validation_passed':True,'RMSE_by_partition':scores,'pooled_MSE':M,'pooled_RMSE':{k:float(np.sqrt(v)) for k,v in M.items()},'conditional_gain_CI95':np.quantile(g,[.025,.975]).tolist(),'new_neural_fits':0,'same400_adaptive_history':True,'current_best':'018','test_or_holdout_read':False,'cost':0,'new_CSV':False,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))};e.write(_repo_path(OUT, 'decision.json'),r);e.write(_repo_path(ROOT, 'reports/cnn-view-mean-151-summary.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','summary',*e.SPLITS,*['validate-'+v for v in e.SPLITS]]);v=p.parse_args().mode
 if v=='prepare':prepare()
 elif v=='summary':summary()
 elif v.startswith('validate-'):validate(v[9:])
 else:run(v)
