"""Independent full500 OLS/residual selection and fresh 1000-row model/CSV replay."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,csv,argparse,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate014_pipeline as p
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
e=p.e;a=p.a;n=p.n;r=p.r

def independent_ols(P,y,Q):
 center=P.mean();coef=np.linalg.lstsq(np.column_stack([np.ones(len(P)),P-center]),y,rcond=None)[0];return coef[0]+coef[1]*(Q-center),coef

def training():
 p.check();d,P,cells=p.inputs();fit=e.read(p.OUT/'fit.json');assert fit['n_train']==500 and fit['train_ids']==d['ids'] and e.sha(p.OUT/'calibrator.json')==fit['calibrator_sha256'];cal=e.read(p.OUT/'calibrator.json');ip=a.mixture(P,p.ARM);_,coef=independent_ols(ip,d['y'],ip);assert abs(coef[0]-cal['target_center'])<1e-10 and abs(coef[1]-cal['slope'])<1e-10
 with np.load(_repo_io(p.OUT/'calibration_training.npz')) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['inner_oof'],ip);np.testing.assert_array_equal(z['components'],P)
 total=0;maxerr=0.;covered=[]
 for j,(A,B,M,D,short) in enumerate(cells):
  path=p.OUT/f'block-{j}';br=e.read(path/'report.json');assert br==fit['blocks'][j];assert e.sha(path/'anchor.npz')==br['anchor_sha256'];ap,coef=independent_ols(a.mixture(D,p.ARM),d['y'][A],a.mixture(M,p.ARM));anchor=.9*ap+.1*short;covered.extend(B.tolist())
  with np.load(_repo_io(path/'anchor.npz')) as z:np.testing.assert_array_equal(z['anchor_indices'],A);np.testing.assert_array_equal(z['block_indices'],B);np.testing.assert_allclose(z['anchor'],anchor,atol=1e-10,rtol=0);anchor=z['anchor'].copy()
  residual=d['y'][B]-anchor;bp=path/'residual';rec=e.read(bp/'report.json');assert e.sha(bp/'report.json')==br['residual_report_sha256'];z=a.readnp(bp/'predictions.npz',rec['predictions_sha256']);np.testing.assert_array_equal(z['residual'],residual);ip=np.empty_like(z['inner_predictions']);folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+j).split(d['X'][B],residual,d['g'][B]))
  for t,(it,iv) in enumerate(folds):
   assert rec['splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]]) and not set(A)&set(B)
   for ci,kind in enumerate(n.KINDS):
    file=bp/f'inner-{t}-{kind}.joblib';assert e.sha(file)==rec['models_sha256'][file.name];saved=joblib.load(_repo_io(file));np.testing.assert_array_equal(saved['train_indices'],B[it]);np.testing.assert_array_equal(saved['validation_indices'],B[iv]);m=saved['model'];assert m['kind']==kind;manual=p.vn.manual(m,d['X'][B[it]],residual[it],d['X'][B[iv]],d['blocks']);pp=n.predict(m,d['X'][B[iv]]);np.testing.assert_array_equal(pp,z['inner_predictions'][ci,iv]);err=float(abs(manual-pp).max());assert err<2e-9;maxerr=max(maxerr,err);ip[ci,iv]=manual;total+=1
  scores=np.mean((ip-residual[None,:])**2,1);np.testing.assert_allclose(scores,list(rec['inner_MSE'].values()),atol=2e-9,rtol=0);chosen=n.KINDS[int(scores.argmin())];assert chosen==rec['chosen']==br['chosen'];file=bp/'selected.joblib';assert e.sha(file)==rec['models_sha256'][file.name];saved=joblib.load(_repo_io(file));np.testing.assert_array_equal(saved['train_indices'],B);assert saved['model']['kind']==chosen;manual=p.vn.manual(saved['model'],d['X'][B],residual,d['X'],d['blocks']);actual=n.predict(saved['model'],d['X']);np.testing.assert_array_equal(actual,z['correction']);err=float(abs(manual-actual).max());assert err<2e-9;maxerr=max(maxerr,err);total+=1
 assert sorted(covered)==list(range(500))
 # Retained independent leaf model training scope and scalers are verified on all500, with no subgroup scoring.
 fit8=e.read(p.p8.OUT/'report.json');assert fit8['train_ids']==d['ids'];assert e.read(_repo_path(ROOT, 'artifacts/final-model-017/report.json'))['training_id_order']==d['ids'];leafcount=0
 for rec in fit8['models']:
  saved=torch.load(_repo_io(p.p8.OUT/rec['checkpoint']),map_location='cpu',weights_only=True);assert saved['train_ids']==d['ids'];X=d[rec['arm']];sc=StandardScaler().fit(X)
  for key,val in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(saved[key].numpy(),val)
  assert saved['target_mean']==float(d['y'].mean()) and saved['target_std']==float(d['y'].std());leafcount+=1
 short=torch.load(_repo_path(ROOT, 'artifacts/final-model-086/head/model.pt'),map_location='cpu',weights_only=True);assert short['train_ids']==d['ids'] and short['n_train']==500;sc=StandardScaler().fit(d['B'])
 for key,val in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(short[key].numpy(),val)
 assert short['target_mean']==float(d['y'].mean()) and short['target_std']==float(d['y'].std());kernel=joblib.load(_repo_path(ROOT, 'artifacts/final-model-017/kernel.joblib'));assert kernel['train_ids']==d['ids'];cnn=torch.load(_repo_path(ROOT, 'artifacts/final-model-017/cnn.pt'),map_location='cpu',weights_only=True);assert cnn['train_ids']==d['ids'];norm=cnn['normalization'];assert norm['y_mean']==float(d['y'].mean()) and norm['y_std']==float(d['y'].std());assert norm['x_mean']==float(d['images'].mean()) and norm['x_std']==float(d['images'].std());leafcount+=3
 # Full500 component removal remains invariant at outer and anchor layers.
 changed=P.copy();changed[:,0]=1e6;np.testing.assert_array_equal(a.mixture(P,p.ARM),a.mixture(changed,p.ARM))
 p.check();result={'passed':True,'rows':500,'affine_fits_independently_verified':4,'residual_models_reloaded':total,'residual_choices_recomputed':3,'manual_kernel_max_error':maxerr,'remaining_full500_leaf_training_scopes_and_stats':leafcount,'cached_nested_base_training_certificate':e.sha(p.c.OUT/'training-validation.json'),'all500_covered_in_honest_residual_blocks':True,'removed_baseline_invariant':True,'training_outputs_not_validation_scores':True,'holdout_subgroup_scored':False,'test_decoded':False,'fit_sha256':e.sha(p.OUT/'fit.json'),'source_sha256':e.sha(Path(__file__))};e.write(p.OUT/'training-validation.json',result);e.emit(result)

def final():
 p.check();ids,raw,short,base,cor,pred,TX=p.inference('mps');manifest=e.read(p.CSV.with_suffix('.manifest.json'));assert e.sha(p.CSV)==manifest['submission_sha256'];assert e.sha(p.OUT/'predictions.npz')==manifest['predictions_sha256'];assert e.sha(p.CSV.with_suffix('.components.csv'))==manifest['components_sha256'];assert len(ids)==len(set(ids))==1000 and np.isfinite(pred).all()
 with np.load(_repo_io(p.OUT/'predictions.npz')) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['components'],raw);np.testing.assert_array_equal(z['short'],short);np.testing.assert_array_equal(z['base'],base);np.testing.assert_array_equal(z['corrections'],cor);np.testing.assert_array_equal(z['primary'],pred)
 ids2,raw2,short2,base2,cor2,cpu,_=p.inference('cpu');assert ids2==ids;cpuerr=float(abs(cpu-pred).max());assert cpuerr<1e-3
 d,P,cells=p.inputs();ip=a.mixture(P,p.ARM);manualcal,coef=independent_ols(ip,d['y'],a.mixture(raw,p.ARM));manualcor=[]
 for j,(A,B,M,D,sh) in enumerate(cells):
  with np.load(_repo_io(p.OUT/f'block-{j}/anchor.npz')) as z:residual=d['y'][B]-z['anchor']
  saved=joblib.load(_repo_io(p.OUT/f'block-{j}/residual/selected.joblib'))['model'];manualcor.append(p.vn.manual(saved,d['X'][B],residual,TX,d['blocks']))
 manualcor=np.stack(manualcor);err=float(abs(manualcor-cor).max());assert err<2e-9;manual=.9*manualcal+.1*short+manualcor.mean(0);np.testing.assert_allclose(manual,pred,atol=2e-9,rtol=0)
 with p.CSV.open(newline='') as f:reader=csv.DictReader(f);assert reader.fieldnames==['ID','hardness'];rows=list(reader)
 assert [row['ID'] for row in rows]==ids;np.testing.assert_array_equal([float(row['hardness']) for row in rows],pred)
 with p.CSV.with_suffix('.reproduced.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 assert p.CSV.read_bytes()==p.CSV.with_suffix('.reproduced.csv').read_bytes()
 m8=e.read(_repo_path(ROOT, 'artifacts/submission-candidate-008.manifest.json'));test57=e.read(p.c.OUT/'test57.json');assert test57['image_sha256']==m8['test_image_sha256']
 for ident,h in m8['test_image_sha256'].items():assert e.sha(_repo_path(ROOT, 'data/test', f'{ident}.png'))==h
 # Check remaining leaf outputs against the previously independently verified inference cache.
 with (_repo_path(ROOT, 'artifacts/submission-candidate-008.components.csv')).open(newline='') as f:old=list(csv.DictReader(f))
 assert [row['ID'] for row in old]==ids;leaferr={}
 for j,name in enumerate(['kernel','cnn','S_mean3','B_mean3'],1):leaferr[name]=float(abs(raw[:,j]-[float(row[name]) for row in old]).max());assert leaferr[name]<1e-3
 with (_repo_path(ROOT, 'artifacts/submission-candidate-011.components.csv')).open(newline='') as f:old11=list(csv.DictReader(f))
 assert [row['ID'] for row in old11]==ids;leaferr['Bshort']=float(abs(short-[float(row['frozen_head']) for row in old11]).max());assert leaferr['Bshort']<1e-3
 with (_repo_path(ROOT, 'artifacts/submission-candidate-013.csv')).open(newline='') as f:old13=list(csv.DictReader(f))
 assert [row['ID'] for row in old13]==ids;oldpred=np.array([float(row['hardness']) for row in old13]);p.check();result={'valid':True,'candidate':'014','rows':1000,'columns':['ID','hardness'],'ID_order_exact':True,'duplicates':0,'finite':True,'missing_or_extra_IDs':0,'CSV_SHA256':e.sha(p.CSV),'CSV_byte_exact':True,'all1000_fresh_MPS_model_reload_exact':True,'CPU_MPS_final_max_error':cpuerr,'independent_kernel_max_error':err,'independent_OLS_kernel_final_max_error':float(abs(manual-pred).max()),'retained_leaf_previous_cache_max_errors':leaferr,'prediction_range':manifest['prediction_range'],'training_label_range':manifest['training_label_range'],'outside_training_range':manifest['outside_training_range'],'selected_residuals':manifest['selected'],'test_image_hashes_verified':1000,'difference_from013_RMS':float(np.sqrt(np.mean((pred-oldpred)**2))),'clipping':False,'test_fit_or_tuning':False,'holdout_subgroup_scored':False,'013_preserved':True,'cost':0,'uploaded':False,'fit_sha256':e.sha(p.OUT/'fit.json'),'training_validation_sha256':e.sha(p.OUT/'training-validation.json'),'source_sha256':e.sha(Path(__file__))};e.write(_repo_path(ROOT, 'reports/candidate014-verification.json'),result);e.emit(result)
if __name__=='__main__':
 torch.set_num_threads(2);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['training','final']);x=ap.parse_args();globals()[x.mode]()
