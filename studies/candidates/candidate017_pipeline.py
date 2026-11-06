"""Locked140 adjacency-bank extension of015, no curvature adjustment."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,csv,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import spatial_joint_140 as q
import candidate015_pipeline as p
import dinob_062 as backbone
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
e=q.e;a=q.a;OUT=_repo_path(ROOT, 'artifacts/final-model-141');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-017.csv')

def prepare():
 p.check();q.check();decision=e.read(q.OUT/'decision.json');assert decision['advance'] and decision['passed_validation'];assert e.read(q.OUT/'extra-validation.json')['passed'];OUT.mkdir(exist_ok=False);assert not CSV.exists()
 names=['candidate017_pipeline.py','spatial_joint_140.py','validate_spatial_joint_extra_140.py','artifacts/spatial-joint-140/decision.json','artifacts/spatial-joint-140/extra-validation.json','artifacts/spatial-joint-140/features.npz','artifacts/spatial-joint-140/extraction.json','artifacts/final-model-134/fit.json','artifacts/final-model-134/calibrator.json','artifacts/final-model-134/training-validation.json']
 names += [f'artifacts/spatial-joint-140/validation-{s}.json' for s in e.SPLITS]
 rec={'candidate':'017','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'recipe':'Exactly015 affine/raw/shortB; physics57+adjacent528 in samehonestB selectors; fixed32projection/nearestneighbor/none,Ridge1,.1,RBF1,.1; mean3corrections. No quadratic137.','development':decision,'full500':'Reuse unchanged015 independentlyverified base/nestedheads and4affines, refit48augmentedresidualmodels. Reuse400label-freefeatures,extract remaining100 onlyafterlock. Noformerholdoutscore orselection.','test':'Afterfitverification, raw1000DINOtokensforfixedfeatureinference only; no testfit/selection/clipping.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':{f:e.sha(_repo_path(ROOT, f)) for f in q.check()['hashes'] if 'submission-candidate-' in f},'best015_retained':True,'cost':0,'upload':False};e.write(_repo_path(OUT, 'decision.json'),rec);e.emit({'candidate017_locked':True,'new_neural_fits':0,'new_residual_fits':48})

def check():
 p.check();q.check();r=e.read(_repo_path(OUT, 'decision.json'))
 for f,h in {**r['hashes'],**r['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 return r

def feature(ids,split,expected,expected_image_hashes,reuse=True):
 start=time.perf_counter();z=a.readnp(q.OUT/'features.npz',e.read(q.OUT/'extraction.json')['features_sha256']);lookup={v:i for i,v in enumerate(z['ids'].tolist())};P=z['projection'];model=backbone.model().to('mps');before=p.p8.h.state_hash(model.state_dict());values=[];recomputed=0;maxerr=0.;hashes={}
 for j,ident in enumerate(ids):
  image=_repo_path(ROOT, 'data', split, f'{ident}.png');hashes[ident]=e.sha(image);assert hashes[ident]==expected_image_hashes[ident]
  if reuse and ident in lookup and lookup[ident] not in [0,199,399]:out=z['adjacent'][lookup[ident]]
  else:
   xx=torch.from_numpy(p.p8.p6.input_image(image)[None]).to('mps')
   with torch.inference_mode():
    tok,cls=model.get_intermediate_layers(xx,n=[11],return_class_token=True,norm=True)[0];desc=torch.cat([cls,tok.mean(1),tok.std(1,correction=0)],1).cpu().numpy()[0];np.testing.assert_array_equal(desc,expected[j,-2304:]);zz=(tok.cpu().numpy()[0].astype(float)@P).reshape(36,36,32);_,out=q.stats(zz)
   recomputed+=1
   if reuse and ident in lookup:err=float(abs(out-z['adjacent'][lookup[ident]]).max());maxerr=max(maxerr,err);np.testing.assert_array_equal(out,z['adjacent'][lookup[ident]])
  values.append(out)
  if (j+1)%100==0:e.emit({'feature_split':split,'completed':j+1,'recomputed':recomputed,'seconds':time.perf_counter()-start})
 assert before==p.p8.h.state_hash(model.state_dict()) and all(v.grad is None for v in model.parameters());del model;torch.mps.empty_cache();X=np.array(values);assert X.shape==(len(ids),528) and np.isfinite(X).all()
 with (_repo_path(OUT, f'{split}-adjacent.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X)
 rec={'sha256':e.sha(_repo_path(OUT, f'{split}-adjacent.npz')),'rows':len(ids),'raw_recomputed':recomputed,'reused_development':len(ids)-recomputed,'development_recompute_max_error':maxerr,'pooled_B_descriptor_exact':True,'image_sha256':hashes,'frozen_weights_unchanged':True,'seconds':time.perf_counter()-start,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};e.write(_repo_path(OUT, f'{split}-adjacent.json'),rec);return X

def data():
 d,P,cells=p.inputs();z=a.readnp(_repo_path(OUT, 'train-adjacent.npz'),e.read(_repo_path(OUT, 'train-adjacent.json'))['sha256']);assert z['ids'].tolist()==d['ids'];d['X']=np.column_stack([d['X'],z['X']]);d['blocks']=list(d['blocks'])+['joint_token']*528;return d,P,cells

def fit():
 check();start=time.perf_counter();d,P,cells=p.inputs();fit8=e.read(p.p8.OUT/'report.json');feature(d['ids'],'train',d['B'],fit8['training_image_sha256']);d,P,cells=data();records=[]
 for j,(A,B,M,D,short) in enumerate(cells):
  old=p.OUT/f'block-{j}';oldrec=e.read(old/'report.json');anchor=a.readnp(old/'anchor.npz',oldrec['anchor_sha256']);np.testing.assert_array_equal(anchor['block_indices'],B);np.testing.assert_array_equal(anchor['anchor_indices'],A);path=_repo_path(OUT, f'block-{j}');path.mkdir();cor,rec=a.fit_block(d,B,anchor['anchor'],0,j,path/'residual');records.append({'block':j,'chosen':rec['chosen'],'anchor_source':str((old/'anchor.npz').relative_to(ROOT)),'anchor_sha256':oldrec['anchor_sha256'],'residual_report_sha256':e.sha(path/'residual/report.json')});e.emit({'full500_block':j,'chosen':rec['chosen']})
 rec={'n_train':500,'train_ids':d['ids'],'blocks':records,'new_residual_fits':48,'new_neural_fits':0,'unchanged_affines_reused':4,'train_csv_sha256':e.sha(_repo_path(ROOT, 'data/train.csv')),'decision_sha256':e.sha(_repo_path(OUT, 'decision.json')),'train_features_sha256':e.sha(_repo_path(OUT, 'train-adjacent.npz')),'holdout_subgroup_scored':False,'test_read':False,'seconds':time.perf_counter()-start};e.write(_repo_path(OUT, 'fit.json'),rec);check();e.emit({'full500_fit_complete':True,'seconds':rec['seconds']})

def training():
 check();d,P,cells=data();fit=e.read(_repo_path(OUT, 'fit.json'));assert fit['train_ids']==d['ids'];maxerr=0.;count=0;covered=[]
 for j,(A,B,M,D,short) in enumerate(cells):
  rec=fit['blocks'][j];z=a.readnp(_repo_path(ROOT, rec['anchor_source']),rec['anchor_sha256']);manual=.9*q.independent_affine(a.mixture(D,p.ARM),d['y'][A],a.mixture(M,p.ARM))+.1*short;np.testing.assert_allclose(manual,z['anchor'],atol=1e-9,rtol=0);res=d['y'][B]-z['anchor'];covered.extend(B.tolist());bp=_repo_path(OUT, f'block-{j}/residual');br=e.read(bp/'report.json');assert e.sha(bp/'report.json')==rec['residual_report_sha256'];bz=a.readnp(bp/'predictions.npz',br['predictions_sha256']);np.testing.assert_array_equal(bz['residual'],res);ip=np.empty_like(bz['inner_predictions']);folds=list(GroupKFold(3,shuffle=True,random_state=a.r.CVSEED+j).split(d['X'][B],res,d['g'][B]))
  for t,(it,iv) in enumerate(folds):
   assert br['splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]]) and not set(A)&set(B)
   for ci,kind in enumerate(a.n.KINDS):
    f=bp/f'inner-{t}-{kind}.joblib';assert e.sha(f)==br['models_sha256'][f.name];m=joblib.load(_repo_io(f));np.testing.assert_array_equal(m['train_indices'],B[it]);np.testing.assert_array_equal(m['validation_indices'],B[iv]);pred=a.n.predict(m['model'],d['X'][B[iv]]);man=q.vn.manual(m['model'],d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);err=float(abs(pred-man).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(pred,bz['inner_predictions'][ci,iv]);ip[ci,iv]=man;count+=1
  scores=np.mean((ip-res)**2,axis=1);np.testing.assert_allclose(scores,list(br['inner_MSE'].values()),atol=2e-9,rtol=0);assert a.n.KINDS[int(scores.argmin())]==rec['chosen']==br['chosen'];f=bp/'selected.joblib';assert e.sha(f)==br['models_sha256'][f.name];m=joblib.load(_repo_io(f));np.testing.assert_array_equal(m['train_indices'],B);pred=a.n.predict(m['model'],d['X']);man=q.vn.manual(m['model'],d['X'][B],res,d['X'],d['blocks']);err=float(abs(pred-man).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(pred,bz['correction']);count+=1
 assert sorted(covered)==list(range(500));r={'passed':True,'rows':500,'residual_fits_independently_verified':count,'max_error':maxerr,'all500_honest_block_coverage':True,'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'unchanged015_leaf_and_nested_training_certificate':e.sha(p.OUT/'training-validation.json'),'holdout_subgroup_scored':False,'test_read':False};e.write(_repo_path(OUT, 'training-validation.json'),r);e.emit(r)

def test_features():
 check();assert e.read(_repo_path(OUT, 'training-validation.json'))['passed'];m=e.read(_repo_path(ROOT, 'artifacts/submission-candidate-008.manifest.json'));z=a.readnp(_repo_path(ROOT, m['test_B_cache']),m['test_B_sha256']);ids,_=p.c.cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);assert ids==z['ids'].tolist();feature(ids,'test',z['X'],m['test_image_sha256'],False)

def inference(device='mps'):
 check();v=e.read(_repo_path(OUT, 'training-validation.json'));assert v['passed'] and v['fit_sha256']==e.sha(_repo_path(OUT, 'fit.json'));ids,raw,short,base,old,oldpred,X=p.inference(device);z=a.readnp(_repo_path(OUT, 'test-adjacent.npz'),e.read(_repo_path(OUT, 'test-adjacent.json'))['sha256']);assert z['ids'].tolist()==ids;X=np.column_stack([X,z['X']]);cor=[]
 for j in range(3):
  path=_repo_path(OUT, f'block-{j}/residual');r=e.read(path/'report.json');f=path/'selected.joblib';assert e.sha(f)==r['models_sha256'][f.name];cor.append(a.n.predict(joblib.load(_repo_io(f))['model'],X))
 cor=np.stack(cor);pred=base+cor.mean(0);assert np.isfinite(pred).all();return ids,raw,short,base,cor,pred,X

def predict():
 ids,raw,short,base,cor,pred,X=inference();assert len(ids)==len(set(ids))==1000
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),components=raw,short=short,base=base,corrections=cor,primary=pred)
 r={'candidate':'017','rows':1000,'submission_sha256':e.sha(CSV),'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'fit_sha256':e.sha(_repo_path(OUT, 'fit.json')),'train_validation_sha256':e.sha(_repo_path(OUT, 'training-validation.json')),'test_features_sha256':e.sha(_repo_path(OUT, 'test-adjacent.npz')),'sample_submission_sha256':e.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'prediction_range':[float(pred.min()),float(pred.max())],'ID_order_exact':True,'finite':True,'duplicates':0,'selected':[v['chosen'] for v in e.read(_repo_path(OUT, 'fit.json'))['blocks']],'test_fit_or_tuning':False,'clipping':False,'uploaded':False,'cost':0};e.write(CSV.with_suffix('.manifest.json'),r);e.emit(r)

def final():
 ids,raw,short,base,cor,pred,X=inference();man=e.read(CSV.with_suffix('.manifest.json'));assert e.sha(CSV)==man['submission_sha256'];z=a.readnp(_repo_path(OUT, 'predictions.npz'),man['predictions_sha256']);assert z['ids'].tolist()==ids
 for name,arr in [('components',raw),('short',short),('base',base),('corrections',cor),('primary',pred)]:np.testing.assert_array_equal(z[name],arr)
 ids2,_,_,_,_,cpu,_=inference('cpu');assert ids2==ids;err=float(abs(cpu-pred).max());assert err<1e-3;d,P,cells=data();mc=[]
 for j,(A,B,M,D,sh) in enumerate(cells):
  fr=e.read(_repo_path(OUT, 'fit.json'))['blocks'][j];anchor=a.readnp(_repo_path(ROOT, fr['anchor_source']),fr['anchor_sha256'])['anchor'];m=joblib.load(_repo_path(OUT, f'block-{j}/residual/selected.joblib'))['model'];mc.append(q.vn.manual(m,d['X'][B],d['y'][B]-anchor,X,d['blocks']))
 manualbase=.9*q.independent_affine(a.mixture(P,p.ARM),d['y'],a.mixture(raw,p.ARM))+.1*short;manual=manualbase+np.mean(mc,0);diff=float(abs(manual-pred).max());assert diff<2e-9
 with CSV.open(newline='') as f:r=csv.DictReader(f);assert r.fieldnames==['ID','hardness'];rows=list(r)
 assert [v['ID'] for v in rows]==ids;np.testing.assert_array_equal([float(v['hardness']) for v in rows],pred)
 with CSV.with_suffix('.reproduced.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 assert CSV.read_bytes()==CSV.with_suffix('.reproduced.csv').read_bytes();check();r={'valid':True,'candidate':'017','rows':1000,'finite':True,'ID_order_exact':True,'duplicates':0,'CSV_byte_exact':True,'all1000_fresh_MPS_model_reload_exact':True,'CPU_MPS_max_error':err,'independent_affine_kernel_max_error':diff,'CSV_SHA256':e.sha(CSV),'prediction_range':man['prediction_range'],'best015_preserved':True,'holdout_subgroup_scored':False,'test_fit_or_tuning':False,'new_neural_fits':0,'cost':0,'uploaded':False};e.write(_repo_path(ROOT, 'reports/candidate017-verification.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','fit','training','test_features','predict','final']);globals()[parser.parse_args().mode]()
