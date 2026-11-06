"""One preregistered projected-covariance block, development-only frozen backbone."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
from dino_regression_037 import FrozenRegressor
import dino_spatial_039 as ds
import frozen_validation as fv
import token_covariance_056 as f
import test_token_covariance_056 as tests
OUT=_repo_path(ROOT, 'artifacts/token-cov-056');ALPHAS=(52.8,528.,5280.)
def ids_only():return sorted(json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text())['development_folds'])
def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,d in {**p['sources'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==d,n
 official=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
 for n,d in official['official_source_sha256'].items():assert fv.sha(ds.base.SOURCE/n)==d,n
 return p
def prepare():
 OUT.mkdir(exist_ok=False);prior=json.loads((_repo_path(ROOT, 'artifacts/betti-055/plan.json')).read_text());plan={'hypothesis':'Cross-channel covariance of pretrained patch activations may preserve texture coactivation lost by per-channel mean/std. One final-block descriptor only.','duplicate_audit':'cnn_texture_fusion/cnn_cov_regularized learn CNN projection on scratch features. dino_texture042 uses per-channel horizontal/vertical/diagonal neighboring-token squared differences. dino_multilayer044 uses normalized CLS/mean/std only. No equivalent fixed projected pretrained channel covariance found.','references':['https://www.kaggle.com/competitions/csiro-biomass/writeups/5th-place-solution','https://doi.org/10.1016/j.actamat.2025.121132'],'reference_access':'Kaggle body not readable via tool; DOI403; title confirmed via publisher/institution search. Parent-provided motivation only, no claimed paper reproduction or hardness evidence.','features':'Original official frozen DINOv2ViT-S14 finalblock12 normalized patchtokens, full504bicubic grayscaleRGB/ImageNetnorm as007,1296x384. Final block fixed before results; no layer selection. NumPyPCG64seed20261225 Gaussian384x32 dividedsqrt384, generated before images/labels. Population centered covariance of projected tokens, uppertriangle includingdiag528, no sqrt2offdiag scaling,no whitening,trace normalization,or eigenvalue transform.','math':'Cproj=R.T C384 R, not full384 covariance or uncenteredGram. Covariance insensitive to token permutation after backbone; discards explicit spatial arrangement, higher than second moments. Backbone position encoding/context can still affect individual tokens.','regression':'Single StandardScaler+standardtarget Ridge alpha52.8/528/5280 selected by innerGroupKFold3seed20261225+outerfold. All scaler/target/Ridge fit innertrain/outertrain only. Primary=.9samefold007+.1covariance Ridge; no residual targets.','screen':'Original gain>conditionalbootstrapSD and>=3/5fold gains. Only then same recipe on both existing repartitions. No weights/layer/projection seeds sweep. Diagnostic covariance alone not alternate winner.','diagnostics':'Compare Cproj to R.Tdiag(channelvariance)R computable from existing std; mean/std control reproduced against044. Record residualcorrelation with007. Descriptive foldtrain Ridge(alpha1152) reconstruction from final CLS+mean/std to covariance; no use for feature selection.','scope':'Development400 only; noholdout/test/candidateCSV/upload/install/cost/externaldata. Existing models preserved.','resources':'Imagewise frozen inference; benchmark3MPSimages plus1CPU; require projected400<300s,RSS<2GB,MPSdriver<3GB.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['token_covariance_056.py','test_token_covariance_056.py','token_cov_experiment_056.py','dino_regression_037.py','dino_spatial_039.py','dino_frozen_037.py']},'protected':prior['protected'],'bias':fv.CAVEAT}
 with (_repo_path(OUT, 'projection.npy')).open('xb') as file:np.save(_repo_io(file),f.projection())
 plan['projection_sha256']=fv.sha(_repo_path(OUT, 'projection.npy'));fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps({'prepared':True,'features':528}),flush=True)
def evaluate(m,ident,device,R):
 assert ident in ids_only() and not m.training and not any(p.requires_grad for p in m.parameters())
 with torch.inference_mode():
  tokens=m.forward_features(torch.from_numpy(ds.inputs(ident,'full504')).to(device));p=tokens['x_norm_patchtokens'];assert p.shape==(1,1296,384);desc=torch.cat((tokens['x_norm_clstoken'],p.mean(1),p.std(1,correction=0)),1).cpu().numpy()[0];P=p.cpu().numpy()[0]
 values,C,D=f.describe(P,R);eig=np.linalg.eigvalsh(C);assert eig.min()>-1e-10
 meta={'min_eigenvalue':float(eig.min()),'symmetry_max_error':float(abs(C-C.T).max()),'cross_channel_contribution_relative_frobenius':float(np.linalg.norm(C-D)/max(np.linalg.norm(C),1e-15)),'trace':float(np.trace(C))}
 return values,desc,meta

def benchmark():
 plan=check();fixture=tests.run();fv.write(_repo_path(OUT, 'fixtures.json'),fixture);R=np.load(_repo_path(OUT, 'projection.npy'));np.testing.assert_array_equal(R,f.projection());torch.set_num_threads(2);assert torch.backends.mps.is_available();m=ds.base.model();ids=ids_only()[:3];cpu=evaluate(m,ids[0],'cpu',R)[0];m=m.to('mps');t=time.perf_counter();rows=[evaluate(m,i,'mps',R) for i in ids];seconds=time.perf_counter()-t;X=np.array([r[0] for r in rows]);np.testing.assert_allclose(cpu,X[0],atol=2e-3,rtol=2e-3);replay=evaluate(m,ids[0],'mps',R)[0];np.testing.assert_array_equal(replay,X[0])
 with (_repo_path(OUT, 'benchmark_features.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),X=X)
 report={'seconds3':seconds,'projected400_seconds':seconds/3*400,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'cpu_mps_max_abs_difference':float(abs(cpu-X[0]).max()),'mps_replay_max_error':float(abs(replay-X[0]).max())};report['resource_pass']=bool(report['projected400_seconds']<300 and report['rss_peak_bytes']<2e9 and report['mps_driver_bytes']<3e9);fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)
def extract():
 plan=check();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass'];assert fv.sha(_repo_path(OUT, 'projection.npy'))==plan['projection_sha256'];R=np.load(_repo_path(OUT, 'projection.npy'));torch.set_num_threads(2);m=ds.base.model().to('mps');before={n:p.detach().cpu().clone() for n,p in m.state_dict().items()};ids=ids_only();X=[];control=[];meta=[];hashes={};t=time.perf_counter()
 for j,ident in enumerate(ids):
  a,d,v=evaluate(m,ident,'mps',R);X.append(a);control.append(d);meta.append(v);hashes[ident]=fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))
  if (j+1)%80==0:print(json.dumps({'features_done':j+1,'seconds':time.perf_counter()-t}),flush=True)
 X=np.array(X);control=np.array(control);assert X.shape==(400,528) and np.isfinite(X).all();assert all(torch.equal(before[n],p.detach().cpu()) for n,p in m.state_dict().items()) and all(p.grad is None for p in m.parameters())
 with np.load(_repo_path(OUT, 'benchmark_features.npz')) as z:np.testing.assert_array_equal(X[:3],z['X'])
 src=_repo_path(ROOT, 'artifacts/dino-multilayer-044/development_features.npz');assert fv.sha(src)=='6d26b7d78eaf1d094e829984d5665a38dc11b71c918f108384861259b35c4159'
 with np.load(_repo_io(src)) as z:assert z['ids'].tolist()==ids;delta=float(abs(control-z['X'][:,2]).max());np.testing.assert_allclose(control,z['X'][:,2],atol=2e-4,rtol=0)
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),X=X,names=np.array(f.NAMES))
 report={'shape':list(X.shape),'seconds':time.perf_counter()-t,'feature_sha256':fv.sha(_repo_path(OUT, 'development_features.npz')),'metadata':meta,'image_sha256':hashes,'reference_meanstd_max_difference':delta,'weights_unchanged':True,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};fv.write(_repo_path(OUT, 'extraction.json'),report);check();print(json.dumps({k:v for k,v in report.items() if k not in ('metadata','image_sha256')}),flush=True)
def run(split):
 check()
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 ex=json.loads((_repo_path(OUT, 'extraction.json')).read_text());assert fv.sha(_repo_path(OUT, 'development_features.npz'))==ex['feature_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz')) as z:ids=z['ids'].tolist();X=z['X'].copy()
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz')) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 src=_repo_path(ROOT, 'artifacts/seed-stability-052', split, 'development_oof.csv');verified=json.loads((_repo_path(ROOT, 'reports/seed-052-validation.json')).read_text());assert fv.sha(src)==verified['results'][split]['oof_sha256']
 with src.open() as file:rows=list(csv.DictReader(file))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);fid=np.array([int(r['fold']) for r in rows]);ref=np.array([float(r['candidate007_design']) for r in rows]);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);t=time.perf_counter();p=np.full(400,np.nan);logs=[]
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261225+k).split(X[tr],y[tr],groups[tr]));scores=[]
  for alpha in ALPHAS:
   ip=np.full(len(tr),np.nan)
   for it,iv in inner:
    assert not set(groups[tr[it]])&set(groups[tr[iv]]);m=FrozenRegressor('all_ridge',alpha).fit(X[tr[it]],y[tr[it]]);ip[iv]=m.predict(X[tr[iv]])
   scores.append({'alpha':alpha,'rmse':fv.rmse(y[tr],ip)})
  chosen=min(scores,key=lambda a:a['rmse']);m=FrozenRegressor('all_ridge',chosen['alpha']).fit(X[tr],y[tr]);p[va]=m.predict(X[va]);path=out/f'fold-{k}.joblib';joblib.dump({'model':m,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(path));logs.append({'fold':k,'chosen':chosen,'inner_scores':scores,'inner_splits':[[tr[a].tolist(),tr[b].tolist()] for a,b in inner],'train_rmse':fv.rmse(y[tr],m.predict(X[tr])),'validation_rmse':fv.rmse(y[va],p[va]),'model_sha256':fv.sha(path)})
 pred={'reference007':ref,'cov528':p,'primary007':.9*ref+.1*p};scores={n:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for n,v in pred.items()};new=pred['primary007'];gain=fv.rmse(y,ref)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20261226).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-ref)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));comp={'gain':gain,'fold_gains':fg,'positive_folds':sum(v>0 for v in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'residual_correlation':float(np.corrcoef(y-ref,y-p)[0,1])}
 with (out/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),y=y,fold=fid,**pred)
 report={'split':split,'scores':scores,'comparison':comp,'folds':logs,'seconds':time.perf_counter()-t,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'reference_oof_sha256':fv.sha(src),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>boot.std() and sum(v>0 for v in fg)>=3),'primary':'primary007','comparison':comp})
 check();print(json.dumps({k:v for k,v in report.items() if k!='folds'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','extract','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark,'extract':extract}.get(a.mode,lambda:run(a.mode))()
