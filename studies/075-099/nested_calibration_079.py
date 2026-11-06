"""Preregistered inner3-to-inner5 affine-only crossfit control."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nested_calibration_074 as base
import numpy as np,torch
from sklearn.model_selection import GroupKFold
fv=base.fv;prior=base.prior;h=base.h;SEEDS=base.SEEDS;INNER_SEED=base.INNER_SEED;fit_cell=base.fit_cell
OUT=_repo_path(ROOT, 'artifacts/nested-calibration-079')

def check():
 base.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,s in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==s,n
 return p

def inputs(split):
 d=base.inputs(split);v=json.loads((_repo_path(ROOT, 'reports/nested-calibration-074-validation.json')).read_text());p=base.OUT/split/'development_predictions.npz';assert fv.sha(p)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:
  assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid']);np.testing.assert_array_equal(z['reference008'],d['reference008']);d['reference009']=z['primary'].copy()
 return d

def prepare():
 base.check();pub=json.loads((_repo_path(ROOT, 'reports/candidate009-public-submission.json')).read_text());OUT.mkdir(exist_ok=False)
 p={'hypothesis':'Increasing only the inner crossfit fold count from3 to5 brings calibration-training model size closer to outer/full model size, potentially improving transfer of the unchanged affine correction. 078 suggests model reliability depends on training size but does not prove this is the cause of077failure.',
 'single_primary':'Unchanged009 affine y=a+b*p008; only inner GroupKFold count3->5. No077contrast, nonlinear terms, shrinkage, ratio grid, sweep orPublic tuning. GroupKFold shuffle20270110+outerfold; base.fit_cell unchanged with same offset rule and fixed learners/seeds. All preprocessing and kernel selection nested within eachinnertraining set.',
 'sizes':{'development_outer_train':320,'old_inner_train':[213,214],'new_inner_train':256,'old_gap':[106,107],'new_gap':64,'old_gap_fraction_approx':1/3,'new_gap_fraction':.2,'final_train':500,'old_final_inner_train':[333,334],'new_final_inner_train':400,'old_final_gap':[166,167],'new_final_gap':100},
 'comparison':'Run original and both existing resplits with exactly the same recipe regardless of original result. Compare to verified009 samefold OOF. Candidate eligible only if pooled RMSE strictly improves009 in all3 partitions and all independent leakage/reload checks pass. Fold counts/bootstrap diagnostic only; no posthoc criterion changes.',
 'resource':'CPU2/MPS existingvenv only. First original cell serves as benchmark and is reused exactly once. Require projected75fresh fits <=1800seconds, RSS<3GB,MPSdriver<3GB. Abort before remaining fits on failure. Validation adds reload/refit work; no paid alternative.',
 'cache_policy':'Immutable development400 image/features reused with hashes. No old trained074cells reused: train ID sets have213/214 rather than256rows. Benchmark cell reusable only when exact train/validation/outervalidation IDs and immutable config/source hashes match. No approximate matching.',
 'scope':'Development400 only. Holdout100/test1000 never decoded for this experiment. Final500/test only after locked candidate. Preserve009 and all failure records.',
 'bias':fv.CAVEAT+' Same400 repeatedly used. Fold count changes training subsets, dependence and seed ensemble as well as sample size; improvement would validate this recipe, not establish a pure causal sample-size effect.',
 'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['nested_calibration_079.py','validate_nested_calibration_079.py','nested_calibration_074.py','validate_nested_calibration_074_v2.py']},
 'inputs':{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/nested-calibration-074-validation.json','artifacts/reliability-size-078/report.json','reports/candidate009-public-submission.json','artifacts/nested-calibration-074/development_features.npz','artifacts/nested-calibration-074/extraction.json']},
 'protected':pub['protected_sha256']}
 fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'sizes':p['sizes'],'comparison':p['comparison']}),flush=True)

def fold_splits(d,k):
 tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);splits=list(GroupKFold(5,shuffle=True,random_state=INNER_SEED+k).split(d['S'][tr],d['y'][tr],d['g'][tr]));return tr,va,splits

def benchmark():
 check();torch.set_num_threads(2);d=inputs('original');tr,va,splits=fold_splits(d,0);it,iv=splits[0];fd=_repo_path(OUT, 'original/fold-0');fd.mkdir(parents=True,exist_ok=False);pp,r=fit_cell(d,tr[it],tr[iv],va,0,0,fd/'inner-0')
 bench={'first_inner_seconds':r['seconds'],'projected25_seconds':r['seconds']*25,'projected75_seconds':r['seconds']*75,'rss_peak_bytes':r['rss_peak_bytes'],'mps_driver_bytes':r['mps_driver_bytes'],'first_fit_reused':True,'benchmark_report_sha256':fv.sha(fd/'inner-0/report.json'),'old_trained_cache_matches':0,'reason_no_old_matches':'All oldcellstrain213/214; allnewcellstrain256.'};bench['resource_pass']=bool(bench['projected75_seconds']<=1800 and r['rss_peak_bytes']<3e9 and r['mps_driver_bytes']<3e9);fv.write(_repo_path(OUT, 'benchmark.json'),bench);print(json.dumps(bench),flush=True);assert bench['resource_pass']

def run(split):
 check();torch.set_num_threads(2);bench=json.loads((_repo_path(OUT, 'benchmark.json')).read_text());assert bench['resource_pass'];d=inputs(split);out=_repo_path(OUT, split)
 if split=='original':assert out.is_dir() and not (out/'report.json').exists()
 else:out.mkdir(exist_ok=False)
 y=d['y'];fid=d['fid'];raw=d['reference008'];ref=d['reference009'];new=np.full(400,np.nan);folds=[];start=time.perf_counter()
 for k in range(5):
  tr,va,splits=fold_splits(d,k);fd=out/f'fold-{k}'
  if not (split=='original' and k==0):fd.mkdir()
  ip=np.full(len(tr),np.nan)
  for j,(it,iv) in enumerate(splits):
   cell=fd/f'inner-{j}'
   if split=='original' and k==0 and j==0:
    assert fv.sha(cell/'report.json')==bench['benchmark_report_sha256'];rec=json.loads((cell/'report.json').read_text());assert rec['train_indices']==tr[it].tolist() and rec['validation_indices']==tr[iv].tolist() and rec['outer_validation_indices']==va.tolist();assert rec['offset']==100 and rec['train_ids']==[d['ids'][i] for i in tr[it]]
    for n,s in rec['files_sha256'].items():assert fv.sha(cell/n)==s
    with np.load(_repo_io(cell/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['train_indices'],tr[it]);np.testing.assert_array_equal(z['validation_indices'],tr[iv]);pp=z['ensemble008'].copy()
   else:pp,rec=fit_cell(d,tr[it],tr[iv],va,k,j,cell)
   ip[iv]=pp
  assert np.isfinite(ip).all();center=float(ip.mean());targetcenter=float(y[tr].mean());pc=ip-center;slope=float(pc@(y[tr]-targetcenter)/(pc@pc));intercept=targetcenter-slope*center;assert np.isfinite([slope,intercept]).all();new[va]=targetcenter+slope*(raw[va]-center)
  with (fd/'calibration.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,inner_oof=ip,prediction_center=center,target_center=targetcenter,slope=slope,intercept=intercept,outer_prediction=new[va])
  fr={'outer_fold':k,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'slope':slope,'intercept':intercept,'prediction_center':center,'target_center':targetcenter,'calibration_sha256':fv.sha(fd/'calibration.npz'),'inner_report_sha256':{f'inner-{j}/report.json':fv.sha(fd/f'inner-{j}/report.json') for j in range(5)},'outer_rmse_reference009':fv.rmse(y[va],ref[va]),'outer_rmse_calibrated':fv.rmse(y[va],new[va])};fv.write(fd/'report.json',fr);folds.append(fr);print(json.dumps({'split':split,'completed_outer_folds':k+1,'slope_trainonly':slope,'elapsed_seconds':time.perf_counter()-start}),flush=True)
 comp=prior.old.comparison(y,ref,new,fid)
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=y,fold=fid,reference008=raw,reference009=ref,primary=new)
 r={'split':split,'scores':{n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in [('reference008',raw),('reference009',ref),('primary',new)]},'comparison':comp,'calibration_slopes':[r['slope'] for r in folds],'fold_report_sha256':{f'fold-{k}/report.json':fv.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds':time.perf_counter()-start,'candidate_created':False};fv.write(out/'report.json',r);check();print(json.dumps(r),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark}.get(a.mode,lambda:run(a.mode))()
