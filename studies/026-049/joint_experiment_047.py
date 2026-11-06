"""Joint DINO/morphology regression replacing their existing ensemble branches."""
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
from joint_kernel_047 import JointKernel
from dino_regression_037 import FrozenRegressor
import compact_regression
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/joint-kernel-047');ALPHAS=(.1,1.,10.);KINDS=('additive','interaction')
def inputs():
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();M=z['X'].astype(np.float64);y=z['y'].copy();groups=z['groups'].copy()
 sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());assert len(ids)==len(set(ids))==400 and set(ids)==set(sp['development_folds']);report=json.loads((_repo_path(ROOT, 'artifacts/run-012/report.json')).read_text());blocks=report['blocks'];assert M.shape==(400,26) and len(blocks)==26
 with np.load(_repo_path(ROOT, 'artifacts/dino-multilayer-044/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;D=z['X'].reshape(400,3456).astype(np.float64)
 assert np.isfinite(D).all() and np.isfinite(M).all();return ids,D,M,y,groups,blocks

def prepare():
 OUT.mkdir(exist_ok=False);paths=[*ROOT.glob('artifacts/submission-candidate-006*'),_repo_path(ROOT, 'artifacts/final-model-045/dino-regressor.joblib'),_repo_path(ROOT, 'artifacts/final-model-045/decision.json'),_repo_path(ROOT, 'artifacts/final-model-045/report.json'),_repo_path(ROOT, 'artifacts/dino-multilayer-044/candidate-design.json'),_repo_path(ROOT, 'reports/dino-044-validation.json')];protected={str(p.relative_to(ROOT)):fv.sha(p) for p in paths}
 assert protected['artifacts/submission-candidate-006.csv']=='1067d401ec27b7328a07812c2f4b24d73254eea63ac9e389f9b6fccbb5379ee2'
 plan={'hypothesis':'Current006 blends morphology and pretrained representations as separate predictions. A jointkernel learns their conditional relationship directly on foldtrain labels; interaction allows pretraineddescriptor coefficients to vary smoothly with morphology. No OOF/residual targets used.','features':'ExistingfrozenDINO6/9/12 descriptor3456 and existingcompact26 grain/phase/pore/intercepts/intensity panel; no imagefeatureaddition oraugmentation','kernels':'KD=train-standardizedDINOlinear/p; KM=RBF on fold-minmaxclip/standardized/blockweightedcompact26, gamma=1/medianpositive trainpairdistance; eachkernel train-centered and unitmeantraindiagonal. KI=centeredtrace-normalizedSchurproduct KD*KM.','inner_choices':{'additive':'.5KD+.5KM','interaction':'.25KD+.25KM+.5KI','alpha':ALPHAS},'selection':'Jointkind and alpha chosen together among6fixedconditions by pooledinner3foldRMSE. All scaler,clipbounds,bandwidth,kernelcenter/scale,targetmeanstd refit insideeachinnertrain. SameinnerGroupKFold3shuffle seed20261120+outerfold.','primary_formula':'.75samefold003-.1875samefoldoldMorphKernel+.4375joint =.375oldbaseline+.1875oldCNN+.4375joint. Existing006morph(.1875)+DINO(.25)weightsmerged,not learned/tuned.','control_formula':'.75samefold003+.25samefoldDINO; freshlyrefitRidge matches006exactly. Oldmorphkernel loadedfromverifiedsamefold037 and validationIDs checked.','diagnostic_formula':'.75samefold003+.25joint,forinterpretationonly;noteligiblealternativeifprimaryfails.','screen':'Original primarygain over006>pairedbootstrapSD and>=3/5foldwins. Only then both existingrepartitions. Do not promote diagnostic or tuneweights basedonouterresults.','confirmation':'If all3partitions improve,recordevidence; assessa prospective train-only repartition before full500candidate. Historical400reuse isnot independentvalidation.','stopping':'Stop this family if original gate fails or confirmation inconsistent; no fallback sweeps or manualtailrangeexpansion.','resources':'CachedfeaturesCPU2threads,anticipated<1GBextraRAM,regressionunder1minute; noMPStrainingrequiredforthisstage.','scope':'development400only,noholdoutselection/testtraining/adaptation/Publicinversion,newpackages/externaldata/cost/upload','source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ('joint_kernel_047.py','joint_experiment_047.py','dino_regression_037.py','compact_regression.py')},'input_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ('artifacts/run-012/development_features.npz','artifacts/run-012/report.json','artifacts/dino-multilayer-044/development_features.npz')},'protected':protected}
 fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps(plan),flush=True)
def run(split):
 plan=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in {**plan['source_sha256'],**plan['input_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 ids,D,M,y,groups,blocks=inputs();src=_repo_path(ROOT, 'artifacts/dino-multilayer-044', split, 'development_oof.csv');checked=json.loads((_repo_path(ROOT, 'reports/dino-044-validation.json')).read_text());assert fv.sha(src)==checked['results'][split]['oof_sha256']
 with src.open() as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);fid=np.array([int(r['fold']) for r in rows]);ref003=np.array([float(r['reference003']) for r in rows]);ref006=np.array([float(r['multi3_blend']) for r in rows]);refD=np.array([float(r['multi3']) for r in rows]);priorroot=_repo_path(ROOT, 'artifacts/dino-frozen-037/regression', split);prior=json.loads((priorroot/'report.json').read_text());pred={n:np.full(400,np.nan) for n in ('control_dino','joint','old_kernel')};logs=[];d=_repo_path(OUT, split);d.mkdir(exist_ok=False);start=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);train_ids=[ids[j] for j in tr];validation_ids=[ids[j] for j in va];inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261120+k).split(D[tr],y[tr],groups[tr]));fd=d/f'fold-{k}';fd.mkdir();t=time.perf_counter()
  path=priorroot/f'fold-{k}'/'compact-kernel.joblib';assert fv.sha(path)==prior['folds'][k]['sha256']['compact-kernel.joblib'];km=joblib.load(_repo_io(path));assert km['train_ids']==train_ids and km['validation_ids']==validation_ids;pred['old_kernel'][va]=km['model'].predict(M[va])
  scores=[];cs=[]
  for alpha in ALPHAS:
   pv=np.full(len(tr),np.nan)
   for it,iv in inner:
    assert not set(groups[tr[it]])&set(groups[tr[iv]]);model=FrozenRegressor('all_ridge',alpha*3456).fit(D[tr[it]],y[tr[it]]);pv[iv]=model.predict(D[tr[iv]])
   cs.append({'alpha':alpha,'rmse':fv.rmse(y[tr],pv)})
   for kind in KINDS:
    pv=np.full(len(tr),np.nan)
    for it,iv in inner:
     model=JointKernel(blocks,kind,alpha).fit(D[tr[it]],M[tr[it]],y[tr[it]]);pv[iv]=model.predict(D[tr[iv]],M[tr[iv]])
    scores.append({'kind':kind,'alpha':alpha,'rmse':fv.rmse(y[tr],pv)})
  c=min(cs,key=lambda q:q['rmse']);cm=FrozenRegressor('all_ridge',c['alpha']*3456).fit(D[tr],y[tr]);pred['control_dino'][va]=cm.predict(D[va]);np.testing.assert_allclose(pred['control_dino'][va],refD[va],atol=1e-10,rtol=0)
  selected=min(scores,key=lambda q:q['rmse']);jm=JointKernel(blocks,selected['kind'],selected['alpha']).fit(D[tr],M[tr],y[tr]);pred['joint'][va]=jm.predict(D[va],M[va]);assert np.isfinite(pred['joint'][va]).all()
  for name,model in (('control',cm),('joint',jm)):joblib.dump({'model':model,'train_ids':train_ids,'validation_ids':validation_ids},_repo_io(fd/f'{name}.joblib'))
  np.testing.assert_allclose(joblib.load(_repo_io(fd/'joint.joblib'))['model'].predict(D[va],M[va]),pred['joint'][va],atol=1e-10,rtol=0)
  rec={'fold':k,'train_ids':train_ids,'validation_ids':validation_ids,'inner_splits':[[tr[a].tolist(),tr[b].tolist()] for a,b in inner],'joint_selection':selected,'joint_inner_scores':scores,'control_selection':c,'control_inner_scores':cs,'joint_train_rmse':fv.rmse(y[tr],jm.predict(D[tr],M[tr])),'joint_validation_rmse':fv.rmse(y[va],pred['joint'][va]),'kernel_gamma':jm.gamma,'train_kernel_trace_mean':float(np.trace(jm.kernel_train)/len(tr)),'source_morph_kernel_sha256':fv.sha(path),'model_sha256':{n:fv.sha(fd/f'{n}.joblib') for n in ('joint','control')},'seconds':time.perf_counter()-t};fv.write(fd/'report.json',rec);logs.append(rec);print(json.dumps({'split':split,'fold':k,'selected':selected,'train_rmse':rec['joint_train_rmse'],'validation_rmse':rec['joint_validation_rmse'],'seconds':rec['seconds']}),flush=True)
 pred['control006']=.75*ref003+.25*pred['control_dino'];np.testing.assert_allclose(pred['control006'],ref006,atol=1e-10,rtol=0);pred['primary']=.75*ref003-.1875*pred['old_kernel']+.4375*pred['joint'];pred['diagnostic_dino_replace']=.75*ref003+.25*pred['joint'];pred['reference003']=ref003
 scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in pred.items()};new=pred['primary'];ix=np.random.default_rng(2026100201).integers(0,400,(3000,400));delta=np.sqrt(np.mean((y-ref006)[ix]**2,axis=1))-np.sqrt(np.mean((y-new)[ix]**2,axis=1));gain=fv.rmse(y,ref006)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],ref006[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];comparison={'gain':gain,'fold_gains':fg,'bootstrap_sd':float(delta.std()),'conditional_ci95':np.percentile(delta,[2.5,97.5]).tolist()}
 with (d/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold',*pred]);w.writerows([i,y[j],fid[j],*[p[j] for p in pred.values()]] for j,i in enumerate(ids))
 report={'scores':scores,'comparison':comparison,'folds':logs,'seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'reference_oof_sha256':fv.sha(src)};fv.write(d/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>delta.std() and sum(v>0 for v in fg)>=3),'gain':gain,'bootstrap_sd':float(delta.std()),'positive_folds':sum(v>0 for v in fg),'diagnostic_ineligible':True})
 for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
 print(json.dumps({'split':split,'scores':{n:q['rmse'] for n,q in scores.items()},'comparison':comparison,'seconds':report['seconds'],'rss_peak_bytes':report['rss_peak_bytes']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','original','20261031','20261101'));a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
