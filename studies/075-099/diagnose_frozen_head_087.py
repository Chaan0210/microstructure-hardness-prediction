"""Read-only matched-feature/scaler and residual diagnostics; no new fit or tuning."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import frozen_complement_085 as study
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
fv=study.fv;e=study.e;OUT=_repo_path(ROOT, 'artifacts/frozen-head-diagnostics-087')

def check_best():
 r=json.loads((_repo_path(ROOT, 'reports/candidate011-public-submission.json')).read_text())
 for n,s in r['protected_sha256'].items():assert fv.sha(_repo_path(ROOT, n))==s,n
 return r

def main():
 start=time.perf_counter();study.check();pub=check_best();torch.set_num_threads(2);OUT.mkdir(exist_ok=False);results={};scalers=0
 for split in study.SPLITS:
  d=e.inputs(split);oldsrc=_repo_path(ROOT, 'artifacts/dinob-mlp-063' if split=='original' else 'artifacts/dinob-complement-064', split)
  oldnpz=_repo_path(ROOT, 'artifacts/dinob-complement-064', split, 'development_predictions.npz')
  with np.load(_repo_io(oldnpz),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid']);oldseeds=z['Bseeds'].copy();oldmean=z['Bmean'].copy()
  with np.load(_repo_io(study.OUT/split/'development_predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];new=z['frozen'].copy();ref=z['reference009'].copy();primary=z['primary'].copy()
  oldtrain=[];newtrain=[]
  for k in range(5):
   tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);npz=study.foldpath(split,k)/'predictions.npz';a=torch.load(_repo_io(study.foldpath(split,k)/'model.pt'),weights_only=True,map_location='cpu');sc=StandardScaler().fit(d['B'][tr]);tp=[]
   with np.load(_repo_io(npz),allow_pickle=False) as z:newtrain.extend((z['train_predictions']-d['y'][tr]).tolist())
   for si,seed in enumerate(e.base.SEEDS):
    b=torch.load(_repo_io(oldsrc/f'fold-{k}'/f'baseB-seed-{seed}.pt'),weights_only=True,map_location='cpu');assert b['train_ids']==a['train_ids'] and b['validation_ids']==a['validation_ids'] and b['target_mean']==a['target_mean'] and b['target_std']==a['target_std']
    for n,v in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(b[n].numpy(),v);np.testing.assert_array_equal(a[n].numpy(),v)
    model=e.h.Head(6912);model.load_state_dict(b['model']);p=e.h.predict(model,sc.transform(d['B'][va]),b['target_mean'],b['target_std']);np.testing.assert_array_equal(p,oldseeds[si,va]);tp.append(e.h.predict(model,sc.transform(d['B'][tr]),b['target_mean'],b['target_std']));scalers+=1
   oldtrain.extend((np.mean(tp,0)-d['y'][tr]).tolist())
  y=d['y'];a=y-ref;b=y-new;delta=new-ref;gain_mse=float(np.mean(a*a)-np.mean((y-primary)**2));linear=float(.2*np.mean(a*delta));quadratic=float(.01*np.mean(delta*delta));assert abs(gain_mse-(linear-quadratic))<1e-10;seedvar=float(np.mean(np.var(oldseeds,axis=0)));avg_mse=float(np.mean((oldseeds-y)**2));mean_mse=float(np.mean((oldmean-y)**2));assert abs((avg_mse-mean_mse)-seedvar)<1e-10
  def metrics(p):return {'rmse':fv.rmse(y,p),'bias_prediction_minus_target':float(np.mean(p-y)),'prediction_std':float(p.std()),'residual_correlation_with009':float(np.corrcoef(y-p,a)[0,1])}
  results[split]={'feature_and_scaler_control':'Identical frozen B6912 data, same train/validation IDs, exact scaler mean/scale/variance and target mean/std across old3 and new heads. No representation/input-normalization change.','old_B_mean3':metrics(oldmean),'new_B_single':metrics(new),'old_B_seed_metrics':[metrics(p) for p in oldseeds],'old_B_mean3_train_RMSE':float(np.sqrt(np.mean(np.array(oldtrain)**2))),'new_B_single_train_RMSE':float(np.sqrt(np.mean(np.array(newtrain)**2))),'new_old_prediction_difference_RMS':float(np.sqrt(np.mean((new-oldmean)**2))),'new_old_residual_correlation':float(np.corrcoef(y-new,y-oldmean)[0,1]),'new_complement_MSE_identity':{'observed_MSE_gain':gain_mse,'error_alignment_term':linear,'disagreement_penalty':quadratic},'old_seed_averaging_identity':{'mean_individual_seed_MSE':avg_mse,'mean3_MSE':mean_mse,'MSE_reduction_equal_between_seed_variance':seedvar},'primary011_RMSE':fv.rmse(y,primary),'reference009_RMSE':fv.rmse(y,ref)}
 r={'passed':True,'results':results,'matched_old_new_scalers_verified':scalers,'old_config':e.h.CONFIG,'new_config':e.CFG,'causal_conclusions':{'representation':'Ruled out as the changed input: same frozen B features.084 joint update control worsened, so backbone adaptation is unsupported.','input_normalization':'Ruled out: scalers and target statistics are exactly equal on every matched outer training fold.','regularization_and_training_length':'Old near-interpolating64wide/dropout.25/wd.01/100epoch versus32wide/dropout.5/wd.1/12epoch changes form a confounded recipe bundle. Stored fits cannot identify width,dropout,weightdecay,epochs individually; batch and CPU/MPS/random-stream differences also remain. Higher new train error supports less interpolation, not a causal allocation among these settings.','seed':'Existing old3seed dispersion establishes optimization randomness and ensemble variance reduction for the old recipe. New recipe has onlyone seed, so its seed variance and mean3 usefulness remain untested. A paired change of seed only is the next controlled test; do not assume the old result transfers.'},'duplicate_audit':'052 averages three old S64wide100epoch heads;063/064 averages old B64wide100epoch heads.084/085/086 use exactlyone new32wide12epoch seed per fold/full500. No new-recipe seed average found.057 early stopping was old S recipe and failed; avoid repeating it.','recommended_single_test':'Keep all084/085 recipe settings and .9/.1 weights; add exactlytwo fixed seeds20270222 and20270308 to existing20270215 and equally average allthree new heads. Compare to011 on allthree existing partitions; never select individual seeds or ratios.','public_result_used_for_tuning':False,'selection_bias':'011 recipe was discovered from084 control after seeing original data and all400 rows have been reused. Diagnosis is descriptive, not independent validation or a causal factorial experiment. New seed experiment isolates algorithmic randomness only.','sources_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ['diagnose_frozen_head_087.py','dinob_mlp_head_063.py','dinob_joint_084.py','reports/frozen-complement-085-validation.json','reports/candidate011-public-submission.json']},'protected011_files':len(pub['protected_sha256']),'no_new_training':True,'holdout_test_access':False,'cost':0,'seconds':time.perf_counter()-start};check_best();fv.write(_repo_path(OUT, 'report.json'),r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
