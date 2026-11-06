"""Full500 split lineage, train-local statistics, independent Ridge and CSV reproduction."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,argparse,time,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate012_residual_pipeline as c
import validate_nested_fusion_112 as v112
import validate_residual_physics_113 as vr
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
e=c.e;r=c.r

def training():
 c.check();torch.set_num_threads(2);d=c.data();X=c.getX();fit=e.read(c.OUT/'fit.json');splits=list(GroupKFold(3,shuffle=True,random_state=e.cal.INNER_SEED).split(d['S'],d['y'],d['g']));maxcnn=0.;maxridge=0.;count=0;start=time.perf_counter();covered=[]
 for j,(A,B) in enumerate(splits):
  covered.extend(B.tolist());path=c.OUT/f'block-{j}';br=e.read(path/'report.json');assert br==fit['blocks'][j];assert e.sha(path/'anchor.npz')==br['anchor_sha256'];assert not set(A)&set(B) and set(A)|set(B)==set(range(500)) and not set(d['g'][A])&set(d['g'][B]);offset=100*(j+1)
  with np.load(_repo_io(path/'anchor.npz'),allow_pickle=False) as z:a={n:z[n].copy() for n in z.files}
  np.testing.assert_array_equal(a['train_indices'],A);np.testing.assert_array_equal(a['residual_indices'],B);p008,err=v112.validate_deep(d,c.p9.OUT/f'inner-{j}',A,B,np.array([],dtype=int),0,j);np.testing.assert_array_equal(p008,a['p008']);maxcnn=max(maxcnn,err);nested=np.full(len(A),np.nan)
  inner=list(GroupKFold(3,shuffle=True,random_state=e.cal.INNER_SEED+offset).split(d['S'][A],d['y'][A],d['g'][A]))
  for ell,(it,iv) in enumerate(inner):
   assert e.sha(path/f'deep-{ell}/report.json')==br['deep_report_sha256'][f'deep-{ell}/report.json'];p,err=v112.validate_deep(d,path/f'deep-{ell}',A[it],A[iv],B,offset,ell);nested[iv]=p;maxcnn=max(maxcnn,err)
  np.testing.assert_array_equal(nested,a['inner_oof008']);center=nested.mean();coef=np.linalg.lstsq(np.column_stack([np.ones(len(A)),nested-center]),d['y'][A],rcond=None)[0];np.testing.assert_allclose(a['p009'],coef[0]+coef[1]*(p008-center),atol=1e-10,rtol=0)
  sr=e.read(path/'shortB/report.json');assert e.sha(path/'shortB/report.json')==br['shortB_report_sha256'] and e.sha(path/'shortB/model.pt')==sr['model_sha256'];saved=torch.load(_repo_io(path/'shortB/model.pt'),map_location='cpu',weights_only=True);assert saved['train_indices']==A.tolist() and saved['validation_indices']==B.tolist() and saved['train_ids']==[d['ids'][i] for i in A];sc=StandardScaler().fit(d['B'][A]);assert saved['target_mean']==d['y'][A].mean() and saved['target_std']==d['y'][A].std()
  for key,value in [('mean',sc.mean_),('scale',sc.scale_),('var',sc.var_)]:np.testing.assert_array_equal(saved[key].numpy(),value)
  model=e.Small(sc.mean_,sc.scale_,saved['seed']).to('mps');model.load_state_dict(saved['state']);p=e.small_predict(model,d['B'],B)*saved['target_std']+saved['target_mean'];np.testing.assert_array_equal(p,a['shortB']);np.testing.assert_array_equal(.9*a['p009']+.1*p,a['p011']);del model;torch.mps.empty_cache()
  rp=path/'residual';rr=e.read(rp/'report.json');assert e.sha(rp/'report.json')==br['residual_report_sha256'] and e.sha(rp/'fit_data.npz')==rr['fit_data_sha256'];res=d['y'][B]-a['p011'];blocks=r.load('original')['blocks'];folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+j).split(X[B],res,d['g'][B]));ip=np.full((3,len(B)),np.nan)
  with np.load(_repo_io(rp/'fit_data.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['residual'],res);oldip=z['inner_predictions'].copy();allpred=z['correction'].copy()
  for t,(it,iv) in enumerate(folds):
   assert rr['inner_splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]]) and not set(A)&set(B[iv])
   for ci,kind in enumerate(r.CANDIDATES):
    file=rp/f'inner-{t}-{kind}.joblib';assert e.sha(file)==rr['models_sha256'][file.name];s=joblib.load(_repo_io(file));np.testing.assert_array_equal(s['train_indices'],B[it]);np.testing.assert_array_equal(s['validation_indices'],B[iv]);m=s['model'];p=vr.manual(m,X[B[it]],res[it],X[B[iv]],blocks);np.testing.assert_array_equal(r.predict(m,X[B[iv]]),oldip[ci,iv]);err=float(abs(p-oldip[ci,iv]).max());assert err<2e-9;maxridge=max(maxridge,err);ip[ci,iv]=p;count+=1
  scores=[r.rmse(res,p) for p in ip];np.testing.assert_allclose(scores,list(rr['inner_scores'].values()),atol=1e-10,rtol=0);selected=r.CANDIDATES[int(np.argmin(scores))];assert selected==rr['chosen']==br['selected'];assert e.sha(rp/'selected.joblib')==rr['selected_sha256'];saved=joblib.load(_repo_io(rp/'selected.joblib'));np.testing.assert_array_equal(saved['train_indices'],B);assert saved['train_ids']==[d['ids'][i] for i in B];p=vr.manual(saved['model'],X[B],res,X,blocks);np.testing.assert_allclose(p,allpred,atol=2e-9,rtol=0);np.testing.assert_array_equal(r.predict(saved['model'],X),allpred);count+=1;e.emit({'training_validated_block':j,'selected':selected})
 assert sorted(covered)==list(range(500));c.check();v={'passed':True,'train_rows':500,'all3_honest_anchor_scopes_verified':True,'deep008_reloaded':9,'middle008_reloaded':3,'shortB_reloaded':3,'inner_KRR_selections_recomputed':12,'residual_models_reloaded':count,'inner_residual_choices_independently_recomputed':3,'max_manual_Ridge_error':maxridge,'max_CNN_CPU_MPS_error':maxcnn,'shortB_MPS_reload_exact':True,'all_scalers_targets_trainlocal':True,'all500_used_as_training_only':True,'holdout_subgroup_scored':False,'stored_full500_residual_predictions_are_unscored_training_outputs':True,'fit_sha256':e.sha(c.OUT/'fit.json'),'source_sha256':e.sha(Path(__file__)),'seconds':time.perf_counter()-start};e.write(c.OUT/'training-validation.json',v);e.emit(v)

def final():
 c.check();v=e.read(c.OUT/'training-validation.json');assert v['passed'] and v['fit_sha256']==e.sha(c.OUT/'fit.json');ids,old,cor,pred=c.outputs();d=c.data();X=c.getX();blocks=r.load('original')['blocks'];test=e.read(c.OUT/'test57.json');assert test['fit_frozen_sha256']==v['fit_sha256']
 with np.load(_repo_io(c.OUT/'test57.npz'),allow_pickle=False) as z:TX=z['X'].copy()
 independent=[]
 for j in range(3):
  path=c.OUT/f'block-{j}'
  with np.load(_repo_io(path/'anchor.npz'),allow_pickle=False) as z:B=z['residual_indices'];res=d['y'][B]-z['p011']
  m=joblib.load(_repo_io(path/'residual/selected.joblib'))['model'];independent.append(vr.manual(m,X[B],res,TX,blocks))
 independent=np.stack(independent);err=float(abs(independent-cor).max());assert err<2e-9;np.testing.assert_allclose(old+independent.mean(0),pred,atol=2e-9,rtol=0)
 with c.CSV.open(newline='') as f:reader=csv.DictReader(f);assert reader.fieldnames==['ID','hardness'];rows=list(reader)
 assert [v['ID'] for v in rows]==ids and len(ids)==len(set(ids))==1000;np.testing.assert_array_equal([float(v['hardness']) for v in rows],pred)
 with np.load(_repo_io(c.OUT/'predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['primary'],pred);np.testing.assert_array_equal(z['corrections'],cor)
 with c.CSV.with_suffix('.reproduced.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 assert e.sha(c.CSV)==e.sha(c.CSV.with_suffix('.reproduced.csv'));man=e.read(c.CSV.with_suffix('.manifest.json'));assert e.sha(c.CSV)==man['submission_sha256'] and e.sha(c.CSV.with_suffix('.components.csv'))==man['components_sha256'];assert np.isfinite(pred).all()
 for ident,h in test['image_sha256'].items():assert e.sha(_repo_path(ROOT, 'data/test', f'{ident}.png'))==h
 # Fixedfirst3 re-extractions validate independent saved feature reload; no fit or selection.
 for ix in [0,499,999]:
  xx,n,h=c.one(ids[ix],'test',True);np.testing.assert_array_equal(xx,TX[ix])
 c.check();result={'valid':True,'candidate':'012','study':'113','CSV_SHA256':e.sha(c.CSV),'rows':1000,'ID_order_exact':True,'duplicates':0,'finite':True,'range':man['prediction_range'],'training_label_range':man['training_label_range'],'outside_training_range':man['outside_training_range'],'clipping':False,'reload_exact':True,'independent_test_Ridge_max_error':err,'CSV_byte_exact':True,'test_image_hashes_verified':1000,'fixed3_test_features_reextracted_exact':True,'selected':man['selected'],'correction_RMS':man['correction_RMS'],'test_fit_or_selection':False,'holdout_subgroup_scored':False,'011_preserved':True,'uploaded':False,'cost':0,'training_validation_sha256':e.sha(c.OUT/'training-validation.json'),'source_sha256':e.sha(Path(__file__))};e.write(_repo_path(ROOT, 'reports/candidate012-residual-verification.json'),result);e.emit(result)
if __name__=='__main__':
 torch.set_num_threads(2);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['training','final']);a=ap.parse_args();globals()[a.mode]()
