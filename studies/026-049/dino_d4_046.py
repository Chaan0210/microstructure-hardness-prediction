"""One fixed frozen DINO D4 mean representation; development400 only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np,torch,joblib
from PIL import Image
from sklearn.model_selection import GroupKFold
from dino_regression_037 import FrozenRegressor
import dino_multilayer_044 as multi
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/dino-d4-046');VIEWS=[(k,f) for f in (False,True) for k in range(4)];LAMBDAS=(.1,1.,10.)

def transform(a,k,f):
 b=np.rot90(a,k)
 if f:b=np.fliplr(b)
 return np.ascontiguousarray(b)
def raw(i):
 with Image.open(_repo_path(ROOT, 'data/train', f'{i}.png')) as im:
  assert im.mode=='L' and im.size==(256,256);return np.array(im,dtype=np.uint8)
def image_input(a):
 a=np.asarray(Image.fromarray(a).resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/255
 return np.ascontiguousarray((np.repeat(a[None],3,axis=0)-multi.ds.base.MEAN)/multi.ds.base.STD)[None]
def prepare():
 OUT.mkdir(exist_ok=False)
 paths=[*ROOT.glob('artifacts/submission-candidate-006*'),_repo_path(ROOT, 'artifacts/final-model-045/dino-regressor.joblib'),_repo_path(ROOT, 'artifacts/final-model-045/decision.json'),_repo_path(ROOT, 'artifacts/final-model-045/report.json'),_repo_path(ROOT, 'artifacts/dino-multilayer-044/candidate-design.json'),_repo_path(ROOT, 'reports/candidate006-verification.json'),_repo_path(ROOT, 'reports/dino-044-validation.json')]
 protected={str(p.relative_to(ROOT)):fv.sha(p) for p in paths};assert protected['artifacts/submission-candidate-006.csv']=='1067d401ec27b7328a07812c2f4b24d73254eea63ac9e389f9b6fccbb5379ee2'
 plan={'hypothesis':'Absolute image-axis orientation may be nuisance for scalar hardness. D4 orbit averaging removes DINO orientation sensitivity while retaining the original images relative alignment and lengths. This is a hypothesis,not established labelinvariance.','risk':'D4 discards absolute orientation and handedness in final DINO descriptor. Relativealignment survives rigid image transforms but nonlinear descriptor averaging may still lose anisotropy information. Label/generator may encode absoluteorientation; measure heldout effect rather than assume.','views':VIEWS,'preprocessing':'Native256rotation by0/90/180/270 then optionalhorizontalreflection;no crop/rescalejitter/intensitychange. Same504bicubicRGB/ImageNetnorm;frozen official DINO blocks6/9/12.','primary':'d4mean: float64mean of8fixed3456descriptors; same3456dimensions, no addedviewvariancefeatures. Train/regress on perimageorbitmean.','control':'control006: originalorientationmulti3 descriptor; recreate006samefoldOOF exactly.','diagnostic':'tta_only: samecontrol006trainfit applied to averaged features; linearRidge makes it equal to mean8viewpredictions. Diagnosticnoteligibleforpromotion; no bestview orweights selected.','regression':'StandardScaler+targetmean/std,Ridgealpha=lambda*3456; lambda(.1,1,10),innerGroupKFold3shuffle seed20261120+outerfold;allfittrainfoldonly.','ensemble':'Unchanged .75samefold003+.25DINO for primary/control/diagnostic. 003 remainsnative; completeensemble isNOTguaranteedD4invariant.','screen':'Original5fold primarygain over006>pairedbootstrapSD and>=3/5foldwins;onlythenbothexistingrepartitions. Screeningisacomputegate,not significance. No diagnostic fallback.','scope':'development400only,noholdout/testimagesorlabels,noexternaldata/packages/cost/candidateCSV/upload. Existingbest006fileshashedonlyforpreservation.','limits':'Historicallyreused400andpartitionscarrymodelselectionoptimism; publicscoresonlyrecordedhistory.','source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ('dino_d4_046.py','dino_multilayer_044.py','dino_spatial_039.py','dino_frozen_037.py','dino_regression_037.py')},'protected':protected}
 fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps(plan),flush=True)
def extract():
 plan=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in plan['source_sha256'].items():assert fv.sha(_repo_path(ROOT, n))==h
 official=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
 for n,h in official['official_source_sha256'].items():assert fv.sha(multi.ds.base.SOURCE/n)==h
 ids,_,_=multi.load()
 with np.load(_repo_io(multi.OUT/'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;identity=z['X'].reshape(400,3456).copy()
 # Label-blind unitcheck: all8reorientedinputs generate the same pixelorbit.
 a=raw(ids[0]);orbit={transform(a,*v).tobytes() for v in VIEWS}
 for v in VIEWS:
  b=transform(a,*v);assert {transform(b,*w).tobytes() for w in VIEWS}==orbit;np.testing.assert_array_equal(np.sort(b.ravel()),np.sort(a.ravel()))
 np.testing.assert_array_equal(image_input(a),multi.ds.inputs(ids[0],'full504'))
 torch.set_num_threads(2);m=multi.ds.base.model().to('mps');before={n:v.detach().cpu().clone() for n,v in m.state_dict().items()};X=np.empty((400,8,3456),dtype=np.float32);X[:,0]=identity;start=time.perf_counter()
 with torch.inference_mode():
  identitycheck=multi.feature(m,torch.from_numpy(image_input(a)).to('mps')).cpu().numpy().reshape(3456);np.testing.assert_array_equal(identitycheck,identity[0])
  for j,i in enumerate(ids):
   a=raw(i)
   for vi,v in enumerate(VIEWS[1:],1):X[j,vi]=multi.feature(m,torch.from_numpy(image_input(transform(a,*v))).to('mps')).cpu().numpy().reshape(3456)
   if (j+1)%40==0:print(json.dumps({'rows':j+1,'seconds':time.perf_counter()-start}),flush=True)
 assert np.isfinite(X).all() and all(torch.equal(before[n],v.detach().cpu()) for n,v in m.state_dict().items());assert all(p.grad is None for p in m.parameters())
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),views=X)
 report={'shape':list(X.shape),'seconds':time.perf_counter()-start,'feature_sha256':fv.sha(_repo_path(OUT, 'development_features.npz')),'identity_exact044':True,'pixelorbit_closed':True,'weights_unchanged':True,'image_sha256':{i:fv.sha(_repo_path(ROOT, 'data/train', f'{i}.png')) for i in ids},'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};fv.write(_repo_path(OUT, 'extraction.json'),report);print(json.dumps(report|{'image_sha256':'saved'}),flush=True)
def run(split):
 plan=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in {**plan['source_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 ids,y,groups=multi.load();ex=json.loads((_repo_path(OUT, 'extraction.json')).read_text());assert fv.sha(_repo_path(OUT, 'development_features.npz'))==ex['feature_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;views=z['views'].astype(np.float64)
 panels={'control006':views[:,0],'d4mean':views.mean(1)};src=multi.OUT/split/'development_oof.csv';checked=json.loads((_repo_path(ROOT, 'reports/dino-044-validation.json')).read_text());assert fv.sha(src)==checked['results'][split]['oof_sha256']
 with src.open() as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);fid=np.array([int(r['fold']) for r in rows]);ref003=np.array([float(r['reference003']) for r in rows]);ref006=np.array([float(r['multi3_blend']) for r in rows]);pred={a:np.full(400,np.nan) for a in (*panels,'tta_only')};viewpred=np.full((400,8),np.nan);bins=np.full(400,-1);logs=[];start=time.perf_counter();d=_repo_path(OUT, split);d.mkdir(exist_ok=False)
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261120+k).split(panels['control006'][tr],y[tr],groups[tr]));fd=d/f'fold-{k}';fd.mkdir();q=np.quantile(y[tr],[.2,.8]);bins[va]=np.digitize(y[va],q);log={'fold':k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'inner_splits':[[tr[a].tolist(),tr[b].tolist()] for a,b in inner],'training_target_quantiles20_80':q.tolist(),'models':{}}
  for arm,A in panels.items():
   scores=[]
   for lam in LAMBDAS:
    p=np.full(len(tr),np.nan)
    for it,iv in inner:
     assert not set(groups[tr[it]])&set(groups[tr[iv]]);m=FrozenRegressor('all_ridge',lam*3456).fit(A[tr[it]],y[tr[it]]);p[iv]=m.predict(A[tr[iv]])
    scores.append({'lambda':lam,'rmse':fv.rmse(y[tr],p)})
   chosen=min(scores,key=lambda a:a['rmse'])['lambda'];m=FrozenRegressor('all_ridge',chosen*3456).fit(A[tr],y[tr]);pred[arm][va]=m.predict(A[va]);path=fd/f'{arm}.joblib';joblib.dump({'model':m,'train_ids':log['train_ids'],'validation_ids':log['validation_ids']},_repo_io(path));np.testing.assert_allclose(joblib.load(_repo_io(path))['model'].predict(A[va]),pred[arm][va],atol=1e-10,rtol=0)
   log['models'][arm]={'chosen_lambda':chosen,'inner_scores':scores,'train_rmse':fv.rmse(y[tr],m.predict(A[tr])),'validation_rmse':fv.rmse(y[va],pred[arm][va]),'sha256':fv.sha(path)}
   if arm=='control006':
    viewpred[va]=m.predict(views[va].reshape(-1,3456)).reshape(len(va),8);pred['tta_only'][va]=m.predict(panels['d4mean'][va]);np.testing.assert_allclose(viewpred[va].mean(1),pred['tta_only'][va],atol=1e-10,rtol=0)
  logs.append(log);fv.write(d/f'fold-{k}-report.json',log)
 pred.update({a+'_blend':.75*ref003+.25*p for a,p in list(pred.items())});pred['reference003']=ref003;np.testing.assert_allclose(pred['control006_blend'],ref006,atol=1e-10,rtol=0)
 scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in pred.items()};new=pred['d4mean_blend'];ix=np.random.default_rng(20261202).integers(0,400,(3000,400));delta=np.sqrt(np.mean((y-ref006)[ix]**2,axis=1))-np.sqrt(np.mean((y-new)[ix]**2,axis=1));gain=fv.rmse(y,ref006)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],ref006[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];comparison={'gain':gain,'fold_gains':fg,'bootstrap_sd':float(delta.std()),'conditional_ci95':np.percentile(delta,[2.5,97.5]).tolist()}
 std=viewpred.std(1);diag={'control_DINO_view_std_quantiles':np.percentile(std,[0,25,50,75,90,100]).tolist(),'control_DINO_view_range_quantiles':np.percentile(np.ptp(viewpred,axis=1),[0,50,90,100]).tolist(),'control_DINO_each_view_rmse':[fv.rmse(y,viewpred[:,j]) for j in range(8)],'view_sensitivity_abs006_residual_correlation':float(np.corrcoef(std,abs(y-ref006))[0,1]),'fold_train_target_bins':{name:{'rows':int((bins==b).sum()),'006_rmse':fv.rmse(y[bins==b],ref006[bins==b]),'d4_rmse':fv.rmse(y[bins==b],new[bins==b]),'006_signed_prediction_error':float((ref006-y)[bins==b].mean()),'d4_signed_prediction_error':float((new-y)[bins==b].mean())} for b,name in enumerate(('low20','middle60','high20'))},'meaning':'DINOcomponent sensitivity only;003component leftnative; bins fromeachfoldtrainquantiles;diagnosticnotfeature/weightselection'}
 with (d/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold',*pred]);w.writerows([i,y[j],fid[j],*[a[j] for a in pred.values()]] for j,i in enumerate(ids))
 with (d/'orientation_diagnostic.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),control_view_predictions=viewpred,target_bins=bins)
 report={'scores':scores,'comparison':comparison,'diagnostic':diag,'folds':logs,'seconds':time.perf_counter()-start,'reference_oof_sha256':fv.sha(src),'orientation_sha256':fv.sha(d/'orientation_diagnostic.npz')};fv.write(d/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>delta.std() and sum(v>0 for v in fg)>=3),'primary':'d4mean','gain':gain,'bootstrap_sd':float(delta.std()),'positive_folds':sum(v>0 for v in fg),'tta_only_diagnostic_ineligible':True})
 for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
 print(json.dumps({'split':split,'scores':{n:a['rmse'] for n,a in scores.items()},'comparison':comparison,'diagnostic':diag,'seconds':report['seconds']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','extract','original','20261031','20261101'));a=p.parse_args();{'prepare':prepare,'extract':extract}.get(a.mode,lambda:run(a.mode))()
