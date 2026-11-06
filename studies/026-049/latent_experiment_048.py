"""One fixed supervised4D latent hypothesis, compared against006."""
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
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
from supervised_latent_048 import LatentKernel
from dino_regression_037 import FrozenRegressor
import joint_experiment_047 as inp
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/supervised-latent-048');ALPHAS=(.1,1.,10.)
def prepare():
 OUT.mkdir(exist_ok=False);old=json.loads((inp.OUT/'plan.json').read_text());plan={'hypothesis':'High-dimensional pretrainedrepresentations include label-irrelevant variation. A train-only supervised4D bottleneck may isolate a small number of hardness-relatedlatentfactors better than previousunsupervisedPCA64(252px). This differs from047jointkernel anddoesnotaddmorphologydata.','input':'Unchangedfrozen006 DINO6/9/12descriptor3456; existingdevelopment400cache','pipeline':'FoldStandardScalerX,targetmean/std;PLSRegression4components(scaleFalse,maxiter500,tol1e-6);unwhitened4Dscore RBFgamma=1/medianpositive squaredtraindistance;traincenter+mean-diagonal1;kernelridgealpha(.1,1,10). All transformations refit insideeachinnertrain.','components':4,'rank_selection':'None;fixedbeforeevaluation. No alternate ranksearch.','primary':'.75samefold003+.25PLS4RBF;no learnedensembleweights','diagnostic':'SamefittedPLS4linear_predict only;notpromotionfallback. Shows whether nonlinearity helps aftertarget-awarecompression.','inner':'GroupKFold3shuffle seed20261120+outerfold,chooseRBFalpha by pooledinnerOOFRMSE;PLS itself is refit for everyinnertrain,never outer-precomputed insideinnerCV.','screen':'Originalprimarygain over006>pairedbootstrapSD and>=3/5foldwins;onlythenboth existingrepartitions. Otherwise stopthisfamily;donottune ranks,widths,weightsortails.','limits':'Supervisedcompressioncanoverfit400samples. Historical400reuse remains selectionoptimism; not independentvalidation.','scope':'Development400only;holdout/testnotused;no newpackages/externaldata/cost/candidateCSV/upload','source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ('supervised_latent_048.py','latent_experiment_048.py','joint_kernel_047.py','joint_experiment_047.py','dino_regression_037.py')},'input_sha256':old['input_sha256'],'protected':old['protected']};fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps(plan),flush=True)
def run(split):
 plan=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in {**plan['source_sha256'],**plan['input_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 ids,D,_,y,groups,_=inp.inputs();src=_repo_path(ROOT, 'artifacts/dino-multilayer-044', split, 'development_oof.csv');checked=json.loads((_repo_path(ROOT, 'reports/dino-044-validation.json')).read_text());assert fv.sha(src)==checked['results'][split]['oof_sha256']
 with src.open() as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);fid=np.array([int(r['fold']) for r in rows]);ref003=np.array([float(r['reference003']) for r in rows]);ref006=np.array([float(r['multi3_blend']) for r in rows]);refD=np.array([float(r['multi3']) for r in rows]);d=_repo_path(OUT, split);d.mkdir(exist_ok=False);pred={n:np.full(400,np.nan) for n in ('control','latent','linear_pls_diagnostic')};logs=[];start=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261120+k).split(D[tr],y[tr],groups[tr]));fd=d/f'fold-{k}';fd.mkdir();scores=[];cs=[];t=time.perf_counter()
  for alpha in ALPHAS:
   pv=np.full(len(tr),np.nan);pc=np.full(len(tr),np.nan)
   for it,iv in inner:
    assert not set(groups[tr[it]])&set(groups[tr[iv]]);m=LatentKernel(alpha).fit(D[tr[it]],y[tr[it]]);pv[iv]=m.predict(D[tr[iv]]);c=FrozenRegressor('all_ridge',alpha*3456).fit(D[tr[it]],y[tr[it]]);pc[iv]=c.predict(D[tr[iv]])
   scores.append({'alpha':alpha,'rmse':fv.rmse(y[tr],pv)});cs.append({'alpha':alpha,'rmse':fv.rmse(y[tr],pc)})
  chosen=min(scores,key=lambda q:q['rmse']);chosen_c=min(cs,key=lambda q:q['rmse']);m=LatentKernel(chosen['alpha']).fit(D[tr],y[tr]);c=FrozenRegressor('all_ridge',chosen_c['alpha']*3456).fit(D[tr],y[tr]);pred['latent'][va]=m.predict(D[va]);pred['linear_pls_diagnostic'][va]=m.linear_predict(D[va]);pred['control'][va]=c.predict(D[va]);np.testing.assert_allclose(pred['control'][va],refD[va],atol=1e-10,rtol=0)
  train_ids=[ids[j] for j in tr];val_ids=[ids[j] for j in va]
  for name,model in (('control',c),('latent',m)):joblib.dump({'model':model,'train_ids':train_ids,'validation_ids':val_ids},_repo_io(fd/f'{name}.joblib'))
  np.testing.assert_allclose(joblib.load(_repo_io(fd/'latent.joblib'))['model'].predict(D[va]),pred['latent'][va],atol=1e-10,rtol=0)
  rec={'fold':k,'train_ids':train_ids,'validation_ids':val_ids,'inner_splits':[[tr[a].tolist(),tr[b].tolist()] for a,b in inner],'selected':chosen,'inner_scores':scores,'control_selected':chosen_c,'control_inner_scores':cs,'latent_train_rmse':fv.rmse(y[tr],m.predict(D[tr])),'linear_pls_train_rmse':fv.rmse(y[tr],m.linear_predict(D[tr])),'validation_rmse':fv.rmse(y[va],pred['latent'][va]),'pls_n_iter':m.pls.n_iter_,'gamma':m.gamma,'seconds':time.perf_counter()-t,'model_sha256':{n:fv.sha(fd/f'{n}.joblib') for n in ('control','latent')}};fv.write(fd/'report.json',rec);logs.append(rec);print(json.dumps({'split':split,'fold':k,'selected':chosen,'train_rmse':rec['latent_train_rmse'],'validation_rmse':rec['validation_rmse'],'seconds':rec['seconds']}),flush=True)
 pred.update({n+'_blend':.75*ref003+.25*p for n,p in list(pred.items())});pred['reference003']=ref003;np.testing.assert_allclose(pred['control_blend'],ref006,atol=1e-10,rtol=0);scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in pred.items()};new=pred['latent_blend'];ix=np.random.default_rng(2026100202).integers(0,400,(3000,400));delta=np.sqrt(np.mean((y-ref006)[ix]**2,axis=1))-np.sqrt(np.mean((y-new)[ix]**2,axis=1));gain=fv.rmse(y,ref006)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],ref006[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];comparison={'gain':gain,'fold_gains':fg,'bootstrap_sd':float(delta.std()),'conditional_ci95':np.percentile(delta,[2.5,97.5]).tolist()}
 with (d/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold',*pred]);w.writerows([i,y[j],fid[j],*[p[j] for p in pred.values()]] for j,i in enumerate(ids))
 report={'scores':scores,'comparison':comparison,'folds':logs,'seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'reference_oof_sha256':fv.sha(src)};fv.write(d/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>delta.std() and sum(v>0 for v in fg)>=3),'gain':gain,'bootstrap_sd':float(delta.std()),'positive_folds':sum(v>0 for v in fg),'linear_diagnostic_ineligible':True})
 for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
 print(json.dumps({'split':split,'scores':{n:a['rmse'] for n,a in scores.items()},'comparison':comparison,'seconds':report['seconds'],'rss_peak_bytes':report['rss_peak_bytes']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','original','20261031','20261101'));a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
