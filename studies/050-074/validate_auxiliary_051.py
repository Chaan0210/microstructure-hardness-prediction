"""Reload and scope audit for all predeclared auxiliary-head comparisons."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import auxiliary_head_051 as h
import auxiliary_experiment_051 as e
import frozen_validation as fv

def main():
 t=time.perf_counter();torch.set_num_threads(2);plan=e.verify();results={};reload_errors=[];fit_replay=[]
 for split in ('original','20261031','20261101'):
  ids,X,y,groups,fid,r003,r006=e.inputs(split);report=json.loads((e.OUT/split/'report.json').read_text())
  with (e.OUT/split/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
  assert [r['ID'] for r in rows]==ids;assert np.array_equal(y,[float(r['hardness']) for r in rows]);assert np.array_equal(fid,[int(r['fold']) for r in rows]);pred={a:np.array([float(r[a]) for r in rows]) for a in report['scores']}
  assert np.array_equal(pred['reference006'],r006);assert np.array_equal(pred['reference003'],r003)
  for fold in range(5):
   tr=np.flatnonzero(fid!=fold);va=np.flatnonzero(fid==fold);assert not set(groups[tr])&set(groups[va]);rec=report['folds'][fold];sc,ym,ys,edges,bins=h.preprocess(X[tr],y[tr]);assert rec['bin_edges']==edges.tolist();assert rec['bin_train_counts']==np.bincount(bins,minlength=7).tolist();assert rec['train_ids']==[ids[j] for j in tr];assert rec['validation_ids']==[ids[j] for j in va]
   logs=rec['models'];assert logs['regression_only']['initial_state_sha256']==logs['auxiliary']['initial_state_sha256'];assert logs['regression_only']['batch_order_sha256']==logs['auxiliary']['batch_order_sha256'];assert logs['regression_only']['classifier_max_parameter_change']==0;assert logs['auxiliary']['classifier_max_parameter_change']>0
   for arm,weight in [('regression_only',0.),('auxiliary',.3)]:
    path=e.OUT/split/f'fold-{fold}'/f'{arm}.pt';assert fv.sha(path)==logs[arm]['checkpoint_sha256'];a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);assert a['config']==h.CONFIG;assert a['aux_weight']==weight;assert a['seed']==h.CONFIG['seed']+fold;assert a['train_ids']==rec['train_ids'] and a['validation_ids']==rec['validation_ids']
    np.testing.assert_array_equal(a['scaler_mean'].numpy(),sc.mean_);np.testing.assert_array_equal(a['scaler_scale'].numpy(),sc.scale_);np.testing.assert_array_equal(a['bin_edges'].numpy(),edges);assert a['target_mean']==ym and a['target_std']==ys
    m=h.Head();m.load_state_dict(a['model']);assert h.state_hash(m.state_dict())==logs[arm]['final_state_sha256'];pv=h.predict(m,sc.transform(X[va]),ym,ys);pt=h.predict(m,sc.transform(X[tr]),ym,ys);err=float(abs(pv-pred[arm][va]).max());assert err<1e-8;reload_errors.append(err)
    assert abs(fv.rmse(y[tr],pt)-logs[arm]['train_rmse'])<1e-8;assert abs(fv.rmse(y[va],pv)-logs[arm]['validation_rmse'])<1e-8;assert logs[arm]['curves'][-1]['epoch']==100;assert [q['epoch'] for q in logs[arm]['curves']]==[0,20,40,60,80,100]
    if split=='original' and fold==0:
     xt=torch.tensor(sc.transform(X[tr]),dtype=torch.float32);yt=torch.tensor((y[tr]-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins,dtype=torch.long);replayed,replay=h.fit(xt,yt,bt,a['seed'],weight);delta=max(float((v-a['model'][k]).abs().max()) for k,v in replayed.state_dict().items());assert delta==0;assert replay['batch_order_sha256']==logs[arm]['batch_order_sha256'];fit_replay.append({'arm':arm,'parameter_max_error':delta})
  for arm in ('regression_only','auxiliary'):
   np.testing.assert_array_equal(pred[arm+'_blend'],.75*r003+.25*pred[arm])
  for arm,v in pred.items():assert np.isfinite(v).all();assert abs(fv.rmse(y,v)-report['scores'][arm]['rmse'])<1e-10
  # Fixed diagnostic only: regression-only was the matched control, not a new winner selected from a sweep.
  ctrl=pred['regression_only_blend'];cg=fv.rmse(y,r006)-fv.rmse(y,ctrl);fg=[fv.rmse(y[fid==k],r006[fid==k])-fv.rmse(y[fid==k],ctrl[fid==k]) for k in range(5)];ix=np.random.default_rng(20261128).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r006)[ix]**2,1))-np.sqrt(np.mean((y-ctrl)[ix]**2,1))
  prior_path=_repo_path(ROOT, 'artifacts/dino-multilayer-044', split, 'development_oof.csv')
  with prior_path.open() as f:old=list(csv.DictReader(f))
  ridge=np.array([float(v['multi3']) for v in old]);rc={k:float(np.corrcoef(y-r003,y-v)[0,1]) for k,v in [('ridge006_component',ridge),('regression_only',pred['regression_only']),('auxiliary',pred['auxiliary'])]}
  end={arm:{'mean_train_rmse':float(np.mean([q['models'][arm]['train_rmse'] for q in report['folds']])),'mean_final_validation_class_accuracy':float(np.mean([q['models'][arm]['curves'][-1]['validation_class_accuracy'] for q in report['folds']])),'mean_final_train_class_accuracy':float(np.mean([q['models'][arm]['curves'][-1]['train_class_accuracy'] for q in report['folds']])),'mean_validation_rmse_change_epoch80_to100':float(np.mean([(q['models'][arm]['curves'][-1]['validation_rmse_standardized']-q['models'][arm]['curves'][-2]['validation_rmse_standardized'])*q['target_std'] for q in report['folds']]))} for arm in ('regression_only','auxiliary')}
  results[split]={'scores':{a:s['rmse'] for a,s in report['scores'].items()},'auxiliary_comparisons':report['comparisons'],'regression_control_vs006':{'gain':cg,'fold_gains':fg,'positive_folds':sum(v>0 for v in fg),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist()},'residual_correlation_to003':rc,'endpoint_diagnostics':end,'oof_sha256':fv.sha(e.OUT/split/'development_oof.csv'),'report_sha256':fv.sha(e.OUT/split/'report.json'),'seconds':report['seconds']}
 e.verify()
 out={'valid':True,'n_models':30,'reload_prediction_max_error':max(reload_errors),'full100epoch_replay':fit_replay,'all_foldtrain_scaler_target_bin_fit_verified':True,'paired_initialization_batch_order_verified':True,'protected_best006_unchanged':True,'results':results,'interpretation':{'auxiliary_specific_effect_consistent':all(v['auxiliary_comparisons']['regression_only_blend']['gain']>0 for v in results.values()),'regression_control_improves006_on_all_partitions':all(v['regression_control_vs006']['gain']>0 for v in results.values()),'primary_auxiliary_improves006_on_all_partitions':all(v['auxiliary_comparisons']['reference006']['gain']>0 for v in results.values()),'candidate_prepared':False,'decision':'Auxiliary-specific benefit not confirmed. MLP regression path is a promising fixed-control finding; no retrospective winner promotion or full500/test fitting in this study.'},'caveats':['One seed and reused400 examples/repartitions; conditional bootstrap intervals ignore prior model-selection bias and shared training fits.','Added CE affects global gradient clipping as well as shared representation gradients; this compares the whole fixed loss-training recipe.','Foldtrain equal-frequency bins are not UEP. Frozen DINOv2 single-target head is not the original DINOv3Large multitarget solution.','Only primary-source snippet was directly readable; seven-bin/weight/public-score details were supplied by parent. No external code/data/weights downloaded.'],'seconds':time.perf_counter()-t}
 fv.write(_repo_path(ROOT, 'reports/auxiliary-051-validation.json'),out);fv.write(_repo_path(ROOT, 'reports/auxiliary-051-summary.json'),{k:v for k,v in out.items() if k!='valid'})
 text='Auxiliary7bin study051 completed: auxiliary-specific gain is inconsistent; shared MLP regression is promising.\n\nSplit | best006 | MLP regression blend | auxiliary blend | auxiliary gain vs regression\n'
 for split,v in results.items():
  a=v['scores'];text+=f"{split} | {a['reference006']:.6f} | {a['regression_only_blend']:.6f} | {a['auxiliary_blend']:.6f} | {a['regression_only_blend']-a['auxiliary_blend']:+.6f}\n"
 text+='\nBoth new branches replace only the25% DINO contribution;75% samefold003 remains fixed. Standalone new heads do not consistently beat Ridge, but their residuals are more complementary to003 (see JSON). This is not a learned weight change.\n'
 text+='Auxiliary-specific gains are +.020409,-.049285,-.002576; no consistent benefit, no classifier-weight/bin/epoch sweep. Regression-only controls improve006 by.153819,.267364,.170397 across three reused partitions. This latter result is a promising control finding, not a predeclared auxiliary success; report it for a separately frozen next decision.\n'
 text+='All30 checkpoint reloads, foldtrain-only X/y normalization and7-quantile edges checked. Two originalfold0 arms were completely replayed for100epochs: identical parameters. Validation curves diagnostic only; no early stopping or checkpoint selection.\n'
 text+='Benchmark CPU2threads, two10epoch runs.976s,peak429MB,shared CE gradient confirmed. Actual3partition30model fitting ~21.35s,peak466MB; benchmark projection was conservative because initialization cost dominated short run.\n'
 text+='No holdout/test access, image feature extraction, new installs/weights/data, fees, candidate or submission. Best006 CSV/design/model hashes retained. No running work.\n'
 text+='Source inspiration: https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution . Browser returned no full body; primary-source indexed snippet confirmed SmoothL1+CE, other specifics supplied by parent. Own implementation, no source code copied. Quantile bins are not UEP; no inference of cross-task gains.\n'
 text+='Reproduce using auxiliary_experiment_051.py prepare,benchmark,original,20261031,20261101 then validate_auxiliary_051.py, through .venv/bin/python -I -B. Outputs exclusive-create.\n'
 (_repo_path(ROOT, 'reports/auxiliary-051-results.txt')).write_text(text)
 from datetime import datetime,timezone
 sp=_repo_path(ROOT, 'reports/status.json');s=json.loads(sp.read_text());s.update(status='auxiliary051_completed_no_candidate',latest_completed='reports/auxiliary-051-summary.json',latest_research='reports/auxiliary-051-summary.json',latest_verification='reports/auxiliary-051-validation.json',latest_research_promoted=False,latest_research_oof_rmse=None,latest_research_partition_rmse={},running_work=[],new_submission_candidate=None,current_study_submission_performed=False,current_study_cost=0,updated_at_utc=datetime.now(timezone.utc).isoformat());sp.write_text(json.dumps(s,indent=2))
 print(json.dumps({k:v for k,v in out.items() if k!='results'}),flush=True)
if __name__=='__main__':main()
