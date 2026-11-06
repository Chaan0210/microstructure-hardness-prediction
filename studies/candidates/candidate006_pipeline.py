"""Frozenmulti-layerDINO final500fit + test1000 prediction; no upload or holdout scoring."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np,torch,joblib
from PIL import Image
from sklearn.model_selection import GroupKFold
import dino_spatial_039 as spatial
import dino_multilayer_044 as multi
from dino_regression_037 import FrozenRegressor
import frozen_validation as fv
import candidate_pipeline as old
OUT=_repo_path(ROOT, 'artifacts/final-model-045')
CSV=_repo_path(ROOT, 'artifacts/submission-candidate-006.csv')
DESIGN=_repo_path(ROOT, 'artifacts/dino-multilayer-044/candidate-design.json')
ALPHAS=(.1,1.,10.)
SOURCES=('candidate006_pipeline.py','dino_multilayer_044.py','dino_spatial_039.py','dino_frozen_037.py','dino_regression_037.py','candidate_pipeline.py','cnn_controls.py')
def protect():
 paths=[_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-003.components.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-003.manifest.json'),DESIGN]
 paths+=list((_repo_path(ROOT, 'artifacts/final-model-017')).glob('*'))
 paths+=[_repo_path(ROOT, n) for n in json.loads((multi.OUT/'plan.json').read_text())['protected']]
 paths+=[p for p in multi.OUT.rglob('*') if p.is_file()]
 paths+=[_repo_path(ROOT, 'reports/dino-044-validation.json'),_repo_path(ROOT, 'reports/dino-044-summary.json')]
 return {str(p.relative_to(ROOT)):fv.sha(p) for p in paths if p.is_file()}
def check(mapping):
 for n,h in mapping.items():assert fv.sha(_repo_path(ROOT, n))==h,n

def input_image(path):
 with Image.open(_repo_io(path)) as im:
  assert im.mode=='L' and im.size==(256,256)
  a=np.asarray(im.resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/255
 return np.ascontiguousarray((np.repeat(a[None],3,axis=0)-spatial.base.MEAN)/spatial.base.STD)
def extract(ids,split):
 torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu';m=spatial.base.model().to(device);before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};values=[];hashes={};start=time.perf_counter()
 for j,ident in enumerate(ids):
  path=_repo_path(ROOT, 'data', split, f'{ident}.png');a=input_image(path);p=multi.feature(m,torch.from_numpy(a[None]).to(device)).reshape(1,3456);values.append(p.cpu().numpy()[0]);hashes[ident]=fv.sha(path)
  if (j+1)%100==0:print(json.dumps({'stage':'features','split':split,'completed':j+1,'seconds':time.perf_counter()-start}),flush=True)
 X=np.stack(values);assert X.shape==(len(ids),3456) and np.isfinite(X).all();assert all(torch.equal(before[k],v.detach().cpu()) for k,v in m.state_dict().items());assert all(p.grad is None for p in m.parameters());del m,before
 if device=='mps':torch.mps.empty_cache()
 return X,hashes,{'device':device,'seconds':time.perf_counter()-start,'weights_unchanged':True}
def summary(a,lo,hi):
 assert np.isfinite(a).all()
 return {'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'std':float(a.std()),'quantiles':np.percentile(a,[0,1,50,99,100]).tolist(),'negative_count':int((a<0).sum()),'below_train_min':int((a<lo).sum()),'above_train_max':int((a>hi).sum()),'beyond_one_training_label_span':int(((a<lo-(hi-lo))|(a>hi+(hi-lo))).sum())}

def fit():
 OUT.mkdir(exist_ok=False);design=json.loads(DESIGN.read_text());assert design['reference_candidate003_sha256']==fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'))
 locked={'design':design,'design_sha256':fv.sha(DESIGN),'source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in SOURCES},'protected':protect(),'inner_seed':20261120,'inner_splits':3,'lambdas':ALPHAS,'fit_scope':'Alltrain500includingformerholdout; no separateholdoutscore,notestread','ensemble':'unchanged003(.75)+frozenmulti3DINO(.25);003alreadyfitonall500;043excluded','source_version':'OriginalpinnedofficialDINOv2alreadyinstalled'};fv.write(_repo_path(OUT, 'decision.json'),locked)
 ids,y=old.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);prior=json.loads((_repo_path(ROOT, 'artifacts/final-model-017/report.json')).read_text());assert prior['n_train']==500 and prior['training_id_order']==ids and prior['train_csv_sha256']==fv.sha(_repo_path(ROOT, 'data/train.csv'))
 sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());groups=np.array([sp['group_by_id'][i] for i in ids]);start=time.perf_counter()
 # The newtrain/test wrapper must be byte-for-byte identical to development preprocessing.
 for i in ids[:3]:np.testing.assert_array_equal(input_image(_repo_path(ROOT, 'data/train', f'{i}.png')),spatial.inputs(i,'full504')[0])
 X,hashes,timing=extract(ids,'train')
 X=X.astype(np.float64);assert hashes==prior['training_image_sha256']
 with np.load(_repo_path(ROOT, 'artifacts/dino-multilayer-044/development_features.npz'),allow_pickle=False) as z:
  indices=np.array([ids.index(i) for i in z['ids'].tolist()]);np.testing.assert_allclose(X[indices],z['X'].reshape(400,3456),rtol=0,atol=2e-4)
 with (_repo_path(OUT, 'train_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X)
 cv=list(GroupKFold(n_splits=3,shuffle=True,random_state=locked['inner_seed']).split(X,y,groups));scores=[]
 for lam in ALPHAS:
  p=np.full(500,np.nan)
  for tr,va in cv:
   assert not set(groups[tr])&set(groups[va]);m=FrozenRegressor('all_ridge',lam*3456).fit(X[tr],y[tr]);p[va]=m.predict(X[va])
  scores.append({'lambda':lam,'alpha':lam*3456,'inner_selection_rmse':fv.rmse(y,p)})
 chosen=min(scores,key=lambda r:r['inner_selection_rmse']);m=FrozenRegressor('all_ridge',chosen['alpha']).fit(X,y);joblib.dump({'model':m,'train_ids':ids,'design_sha256':locked['design_sha256']},_repo_path(OUT, 'dino-regressor.joblib'));assert m.scaler.n_samples_seen_==500;np.testing.assert_allclose(m.scaler.mean_,X.mean(0),atol=1e-10,rtol=0)
 np.testing.assert_allclose(joblib.load(_repo_path(OUT, 'dino-regressor.joblib'))['model'].predict(X),m.predict(X),atol=1e-10,rtol=0)
 check(locked['protected']);check(locked['source_sha256'])
 report={'n_train':500,'train_ids':ids,'train_csv_sha256':fv.sha(_repo_path(ROOT, 'data/train.csv')),'training_image_sha256':hashes,'inner_selection':{'candidates':scores,'selected':chosen,'splits':[[tr.tolist(),va.tolist()] for tr,va in cv],'interpretation':'Training-onlyhyperparameterselection,notfinalvalidation;formerholdoutnotseparatelyevaluated'},'training_label_range':[float(y.min()),float(y.max())],'feature_sha256':fv.sha(_repo_path(OUT, 'train_features.npz')),'model_sha256':fv.sha(_repo_path(OUT, 'dino-regressor.joblib')),'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json')),'extraction':timing,'seconds':time.perf_counter()-start,'test_used':False,'uploaded':False};fv.write(_repo_path(OUT, 'report.json'),report);print(json.dumps({'fit_complete':True,'selected':chosen,'label_range':report['training_label_range'],'seconds':report['seconds']}),flush=True)

def predict():
 assert not CSV.exists() and not CSV.with_suffix('.components.csv').exists() and not CSV.with_suffix('.manifest.json').exists()
 report=json.loads((_repo_path(OUT, 'report.json')).read_text());locked=json.loads((_repo_path(OUT, 'decision.json')).read_text());check(locked['source_sha256']);check(locked['protected']);assert fv.sha(_repo_path(OUT, 'decision.json'))==report['decision_sha256'];assert fv.sha(_repo_path(OUT, 'dino-regressor.joblib'))==report['model_sha256'];m=joblib.load(_repo_path(OUT, 'dino-regressor.joblib'))['model'];before=fv.sha(_repo_path(OUT, 'dino-regressor.joblib'))
 ids,_=old.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);oldmanifest=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-003.manifest.json')).read_text());assert oldmanifest['sample_submission_sha256']==fv.sha(_repo_path(ROOT, 'data/sample_submission.csv'));start=time.perf_counter();X,hashes,timing=extract(ids,'test');X=X.astype(np.float64);assert hashes==oldmanifest['image_sha256']
 with (_repo_path(OUT, 'test_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X)
 with (_repo_path(ROOT, 'artifacts/submission-candidate-003.csv')).open() as f:rows=list(csv.DictReader(f))
 with (_repo_path(ROOT, 'artifacts/submission-candidate-003.components.csv')).open() as f:comps=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==[r['ID'] for r in comps]==ids
 p003=np.array([float(r['hardness']) for r in rows]);dp=m.predict(X);pred=.75*p003+.25*dp
 components={n:np.array([float(r[n]) for r in comps]) for n in ('baseline','cnn','kernel')};np.testing.assert_allclose(p003,.5*components['baseline']+.25*components['cnn']+.25*components['kernel'],atol=1e-10,rtol=0)
 # Diagnose old baseline's individual regressors with verified prediction-only cache.
 cache=_repo_path(ROOT, 'artifacts/test-prediction-features.npz');assert fv.sha(cache)==oldmanifest['baseline_test_feature_cache_sha256']
 for n,h in oldmanifest['model_sha256'].items():assert fv.sha(_repo_path(ROOT, 'artifacts/final-model-017', n))==h
 base=joblib.load(_repo_path(ROOT, 'artifacts/final-model-017/baseline.joblib'))
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids and z['hashes'].tolist()==[hashes[i] for i in ids];XB=z['X']
 for n,reg in base['models'].items():components['baseline_'+n]=reg.predict(XB)
 components.update(candidate003=p003,dino_multi=dp,ensemble006=pred);lo,hi=report['training_label_range'];stats={n:summary(v,lo,hi) for n,v in components.items()}
 np.testing.assert_allclose(joblib.load(_repo_path(OUT, 'dino-regressor.joblib'))['model'].predict(X),dp,rtol=0,atol=1e-10)
 with np.load(_repo_path(OUT, 'train_features.npz'),allow_pickle=False) as z:XT=z['X']
 ztest=m.scaler.transform(X);ztrain=m.scaler.transform(XT);outside=(X<XT.min(0))|(X>XT.max(0));featurediag={'test_max_abs_train_standardized_z':float(abs(ztest).max()),'train_max_abs_z':float(abs(ztrain).max()),'test_outside_train_feature_range_fraction':float(outside.mean()),'test_feature_l2_train_standardized_quantiles':np.percentile(np.linalg.norm(ztest,axis=1),[0,50,95,100]).tolist(),'train_feature_l2_quantiles':np.percentile(np.linalg.norm(ztrain,axis=1),[0,50,95,100]).tolist(),'purpose':'Postfreeze stabilitydiagnostic only;not usedtofit,clip,tune,orselect'}
 assert all(v.shape==(1000,) and np.isfinite(v).all() for v in components.values());assert len(set(ids))==1000
 check(locked['protected']);check(locked['source_sha256']);assert fv.sha(_repo_path(OUT, 'dino-regressor.joblib'))==before
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID',*components]);w.writerows([i,*[v[j] for v in components.values()]] for j,i in enumerate(ids))
 with CSV.open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 manifest={'model':'artifacts/final-model-045','model_sha256':before,'model_report_sha256':fv.sha(_repo_path(OUT, 'report.json')),'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json')),'source_sha256':locked['source_sha256'],'design_sha256':locked['design_sha256'],'submission_sha256':fv.sha(CSV),'components_sha256':fv.sha(CSV.with_suffix('.components.csv')),'test_feature_sha256':fv.sha(_repo_path(OUT, 'test_features.npz')),'sample_submission_sha256':fv.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'rows':1000,'exact_sample_order':True,'duplicate_ids':False,'finite':True,'clipping':False,'test_fit':False,'ensemble_weights':{'candidate003':.75,'dino_multi':.25},'component_summary':stats,'test_feature_stability':featurediag,'test_image_sha256':hashes,'extraction':timing,'seconds':time.perf_counter()-start,'protected':locked['protected'],'uploaded':False};fv.write(CSV.with_suffix('.manifest.json'),manifest);print(json.dumps({'prediction_complete':True,'sha256':manifest['submission_sha256'],'component_ranges':{n:[v['min'],v['max']] for n,v in stats.items()},'seconds':manifest['seconds']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('fit','predict'));a=p.parse_args();fit() if a.mode=='fit' else predict()
