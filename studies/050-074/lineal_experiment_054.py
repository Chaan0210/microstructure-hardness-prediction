"""Single fixed120-feature panel, small inner-selectedRidge andfixedblend."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,joblib
from PIL import Image
from sklearn.model_selection import GroupKFold
from dino_regression_037 import FrozenRegressor
import frozen_validation as fv
import lineal_features_054 as f
import test_lineal_054 as tests
OUT=_repo_path(ROOT, 'artifacts/lineal-054');ALPHAS=(12.,120.,1200.)

def ids_only():
 s=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());return sorted(s['development_folds'])

def image(ident):
 assert ident in ids_only();path=_repo_path(ROOT, 'data/train', f'{ident}.png')
 with Image.open(_repo_io(path)) as im:assert im.size==(256,256) and im.mode=='L';return np.asarray(im,dtype=np.uint8).copy()

def prepare():
 OUT.mkdir(exist_ok=False);protected={str(p.relative_to(ROOT)):fv.sha(p) for p in [*ROOT.glob('artifacts/submission-candidate-006*'),*ROOT.glob('artifacts/submission-candidate-007*'),*(_repo_path(ROOT, 'artifacts/final-model-045')).glob('*'),*(_repo_path(ROOT, 'artifacts/final-model-053')).glob('*'),_repo_path(ROOT, 'artifacts/seed-stability-052/candidate-design.json')] if p.is_file()}
 plan={'hypothesis':'Full-segment probabilities over lengths encode phase connectivity and chord distributions beyond existinginterceptmoments or two-point autocorrelation. Hardness benefit unknown.','references':['https://www.sciencedirect.com/science/article/pii/S0921509307005898','https://www.sciencedirect.com/science/article/abs/pii/S1044580318313743'],'access':'Both directprimaryURLs403; nofulltextretrieved. Implemented user-specifieddefinition,notclaimedpaperreproduction. No externalcode/data downloaded.','panel':'Raw256grayquantiles10/25/50/75/90,mask<=quantileandcomplement,4directions,no denoise orperiodicwrap. Sixlengths2/4/8/16/32/64; directionmean/populationstd gives5*2*6*2=120.','distance_definition':'L=pixel-center endpointdistance. Lattice lagk usesk+1consecutivephasepixels. Horizontal/vertical spacing1; diagonalspacing sqrt2. Adjacentlagprobabilities linearlyinterpolatedtoL/sqrt2, eachnormalizedbyitsownvalidstartcount. Digital approximation; no exactsubpixelclaim.','counting':'Sum over observedphase runlengthr ofmax(r-k,0) /sumoverimage-line lengthsN ofmax(N-k,0). Includesimage-edgeclippedruns butonlywhollyvisiblewindows;doesnotdiscardshortdiagonals exceptwhennoeligiblewindow.','quantile_ties':'np.quantilelinear;mask<=threshold. Eight-bit ties retained,D4invariancepreserved. Nominalphasefractionslargelyremoved butactualoccupancycanexceedq;countsloggeddiagnostically,notaddedfeatures.','existing_overlap':'latent36 usesblackhatboundarythresholdinterceptmoments,notrawgraylevelwhole-segmentcurves. Existingextra granulometry/distance/componentstatsdistinct; no Euler/Betti added.','regression':'OneRidgefamily,StandardScalerX,targetmeanstdinnertrainonly; alpha12/120/1200chosenbysmallinnerGroupKFold3,seed20261221+outerfold,pooledstandaloneinnerRMSE. Noresidualtargets orbaselinein-samplelabels.','primary':'.90samefold007+.10linealRidge. Existingphase/intensityinformationretainedwithin007,notreplaced.','diagnostic':'.90samefold006+.10linealRidge andlinealalone;noteligiblealternatewinner.','screen':'Originalprimaryimproves007,>=3/5folds,andgainexceedspairedbstrapSD. Thisiscomputeallocationrule,notsignificancethreshold. Onlythenconfirmunchangedrecipeonbothknownrepartitions. Stopifnotpassed,nofeature/weight/hyperparametersweep.','scope':'Development400only;noholdout/testdata,submissionCSV,upload,newinstall,cost,topologyfeatures.006/007artifactsprotected.','resource':'Fixturefirst;benchmarkfirst10developmentimages,thenfixed400extractionifprojected<5minutesandRSS<2GB.','bias':'Development400repeatedlyused;conditionalbootstrapdoesnotcorrecthistoricalmodelselection.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['lineal_features_054.py','test_lineal_054.py','lineal_experiment_054.py','dino_regression_037.py']},'protected':protected}
 fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps({'prepared':True,'features':120}),flush=True)

def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,d in {**p['sources'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==d,n
 return p

def benchmark():
 check();fixture=tests.run();assert fixture['passed'];fv.write(_repo_path(OUT, 'fixtures.json'),fixture);ids=ids_only()[:10];t=time.perf_counter();A=[];meta=[]
 for ident in ids:a,m=f.extract(image(ident));A.append(a);meta.append(m)
 elapsed=time.perf_counter()-t;report={'n_images':10,'seconds':elapsed,'per_image_seconds':elapsed/10,'projected400_seconds':elapsed*40,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'fixture_pass':True,'fixture_count':fixture['count']};report['resource_pass']=report['projected400_seconds']<300 and report['peak_rss_bytes']<2e9
 with (_repo_path(OUT, 'benchmark_features.npz')).open('xb') as out:np.savez_compressed(_repo_io(out),ids=np.array(ids),X=np.array(A))
 fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)

def extract():
 check();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass'];ids=ids_only();t=time.perf_counter();X=[];meta=[];hashes={}
 for j,ident in enumerate(ids):
  a,m=f.extract(image(ident));X.append(a);meta.append(m);hashes[ident]=fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))
  if (j+1)%80==0:print(json.dumps({'features_done':j+1,'seconds':time.perf_counter()-t}),flush=True)
 X=np.array(X);assert X.shape==(400,120);assert np.isfinite(X).all()
 with np.load(_repo_path(OUT, 'benchmark_features.npz'),allow_pickle=False) as b:assert b['ids'].tolist()==ids[:10];np.testing.assert_array_equal(b['X'],X[:10])
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as out:np.savez_compressed(_repo_io(out),ids=np.array(ids),X=X,names=np.array(f.NAMES))
 report={'shape':list(X.shape),'feature_sha256':fv.sha(_repo_path(OUT, 'development_features.npz')),'image_sha256':hashes,'metadata':meta,'actual_mask_fraction_quantiles_by_q':np.quantile(np.array([m['mask_fractions'] for m in meta]),[0,.5,1],axis=0).tolist(),'seconds':time.perf_counter()-t,'benchmark_replay_exact':True};fv.write(_repo_path(OUT, 'extraction.json'),report);check();print(json.dumps({k:v for k,v in report.items() if k not in ('image_sha256','metadata')}),flush=True)

def run(split):
 check()
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 ex=json.loads((_repo_path(OUT, 'extraction.json')).read_text());assert fv.sha(_repo_path(OUT, 'development_features.npz'))==ex['feature_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();X=z['X'].copy()
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 src=_repo_path(ROOT, 'artifacts/seed-stability-052', split, 'development_oof.csv');verified=json.loads((_repo_path(ROOT, 'reports/seed-052-validation.json')).read_text());assert fv.sha(src)==verified['results'][split]['oof_sha256']
 with src.open() as inp:rows=list(csv.DictReader(inp))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);fid=np.array([int(r['fold']) for r in rows]);ref007=np.array([float(r['candidate007_design']) for r in rows]);ref006=np.array([float(r['reference006']) for r in rows]);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);p=np.full(400,np.nan);logs=[];t=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261221+k).split(X[tr],y[tr],groups[tr]));scores=[]
  for alpha in ALPHAS:
   ip=np.full(len(tr),np.nan)
   for it,iv in inner:
    assert not set(groups[tr[it]])&set(groups[tr[iv]]);m=FrozenRegressor('all_ridge',alpha).fit(X[tr[it]],y[tr[it]]);ip[iv]=m.predict(X[tr[iv]])
   scores.append({'alpha':alpha,'rmse':fv.rmse(y[tr],ip)})
  chosen=min(scores,key=lambda a:a['rmse']);m=FrozenRegressor('all_ridge',chosen['alpha']).fit(X[tr],y[tr]);p[va]=m.predict(X[va]);path=out/f'fold-{k}.joblib';joblib.dump({'model':m,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(path));np.testing.assert_array_equal(joblib.load(_repo_io(path))['model'].predict(X[va]),p[va]);logs.append({'fold':k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'inner_splits':[[tr[a].tolist(),tr[b].tolist()] for a,b in inner],'inner_scores':scores,'chosen':chosen,'train_rmse':fv.rmse(y[tr],m.predict(X[tr])),'validation_rmse':fv.rmse(y[va],p[va]),'model_sha256':fv.sha(path)})
 pred={'lineal':p,'primary007':.9*ref007+.1*p,'diagnostic006':.9*ref006+.1*p,'reference007':ref007,'reference006':ref006};scores={n:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for n,v in pred.items()};new=pred['primary007'];gain=fv.rmse(y,ref007)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],ref007[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20261222).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-ref007)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));comparison={'gain':gain,'fold_gains':fg,'positive_folds':sum(v>0 for v in fg),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'bootstrap_sd':float(boot.std()),'lineal_007_residual_correlation':float(np.corrcoef(y-ref007,y-p)[0,1])}
 # Development-only JSON/NPZ, no new submission CSV.
 with (out/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),y=y,fold=fid,**pred)
 report={'split':split,'scores':scores,'comparison':comparison,'folds':logs,'seconds':time.perf_counter()-t,'reference_oof_sha256':fv.sha(src),'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>boot.std() and sum(v>0 for v in fg)>=3),'gain':gain,'bootstrap_sd':float(boot.std()),'positive_folds':sum(v>0 for v in fg),'primary':'primary007','diagnostic_ineligible':True})
 check();print(json.dumps({k:v for k,v in report.items() if k!='folds'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','extract','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark,'extract':extract}.get(a.mode,lambda:run(a.mode))()
