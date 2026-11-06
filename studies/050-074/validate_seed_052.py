"""Verify allseed checkpoints before freezing candidate007 design."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import seed_stability_052 as s
import auxiliary_head_051 as h
import auxiliary_experiment_051 as e
import frozen_validation as fv

def main():
 t=time.perf_counter();torch.set_num_threads(2);plan=s.check();results={};replays=[];maxerr=0.;nmodels=0
 for split in s.SPLITS:
  ids,X,y,groups,fid,r003,r006=e.inputs(split);report=json.loads((s.OUT/split/'report.json').read_text())
  with (s.OUT/split/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
  assert [v['ID'] for v in rows]==ids;assert np.array_equal(y,[float(v['hardness']) for v in rows]);assert np.array_equal(fid,[int(v['fold']) for v in rows]);pred=np.stack([[float(v[f'seed_{base}']) for v in rows] for base in s.SEEDS]);trainres=[]
  for fold in range(5):
   tr=np.flatnonzero(fid!=fold);va=np.flatnonzero(fid==fold);assert not set(groups[tr])&set(groups[va]);sc,ym,ys,edges,bins=h.preprocess(X[tr],y[tr]);trainpred=[]
   for k,base in enumerate(s.SEEDS):
    r=report['folds'][fold]['models'][str(base)];path=_repo_path(ROOT, r['checkpoint']);assert fv.sha(path)==r['checkpoint_sha256'];a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);assert a['seed']==base+fold and a['aux_weight']==0;assert a['config']==h.CONFIG;assert a['train_ids']==[ids[j] for j in tr];assert a['validation_ids']==[ids[j] for j in va];np.testing.assert_array_equal(a['scaler_mean'].numpy(),sc.mean_);np.testing.assert_array_equal(a['scaler_scale'].numpy(),sc.scale_);assert a['target_mean']==ym and a['target_std']==ys;model=s.restore(a);pv=h.predict(model,sc.transform(X[va]),ym,ys);err=float(abs(pv-pred[k,va]).max());maxerr=max(maxerr,err);assert err==0;trainpred.append(h.predict(model,sc.transform(X[tr]),ym,ys));nmodels+=1
    assert r['curves'][-1]['epoch']==100 and r['classifier_max_parameter_change']==0
    if split=='original' and fold==0 and base!=s.SEEDS[0]:
     replay,_=h.fit(torch.tensor(sc.transform(X[tr]),dtype=torch.float32),torch.tensor((y[tr]-ym)/ys,dtype=torch.float32),torch.tensor(bins,dtype=torch.long),base,0.);delta=max(float(abs(v-a['model'][key]).max()) for key,v in replay.state_dict().items());assert delta==0;replays.append({'seed':base,'max_parameter_error':delta})
   trainres.extend((np.mean(trainpred,axis=0)-y[tr]).tolist())
  mean=pred.mean(0);candidate=.75*r003+.25*mean;assert np.isfinite(pred).all();np.testing.assert_array_equal(mean,[float(v['mean3']) for v in rows]);np.testing.assert_array_equal(candidate,[float(v['candidate007_design']) for v in rows]);score=fv.rmse(y,candidate);assert abs(score-report['mean3_blend_rmse'])<1e-12;assert abs(float(np.sqrt(np.mean(np.array(trainres)**2)))-report['mean3_train_rmse_pooled'])<1e-12;gain=fv.rmse(y,r006)-score;assert gain==report['comparison']['gain']
  delta=(y-r006)**2-(y-candidate)**2;order=np.argsort(-delta,kind='stable')
  for stress in report['influence_stress']:
   k=stress['removed_n'];assert stress['removed_ids']==[ids[j] for j in order[:k]];keep=np.ones(400,bool);keep[order[:k]]=False;assert abs(fv.rmse(y[keep],r006[keep])-fv.rmse(y[keep],candidate[keep])-stress['remaining_rmse_gain'])<1e-12
  results[split]={k:v for k,v in report.items() if k!='folds'};results[split].update(oof_sha256=fv.sha(s.OUT/split/'development_oof.csv'),report_sha256=fv.sha(s.OUT/split/'report.json'))
 s.check();passed=all(v['comparison']['gain']>0 for v in results.values());validation={'valid':True,'gate_all_three_improve006':passed,'model_count':nmodels,'new_model_count':30,'reused_model_count':15,'max_reload_error':maxerr,'full_epoch_replays':replays,'three_seeds_equal_weight_all_retained':True,'foldtrain_only_transforms_verified':True,'results':results,'influence_caveat':'Removing greatestSSEbenefitrows is worst-case forSSEgain, not mathematically worst-case RMSEgap;diagnostic only. Top10 contribute~74-78%ofnetSSEgain, but remainingRMSEgainstayspositive inall3.','selection_bias':'Samehistoricallyreused400,notnewindependentvalidation; one3seedsetnotproofagainstfuturetestshift.','protected_unchanged':True,'seconds':time.perf_counter()-t};fv.write(_repo_path(ROOT, 'reports/seed-052-validation.json'),validation);fv.write(_repo_path(ROOT, 'reports/seed-052-summary.json'),{k:v for k,v in validation.items() if k not in ('valid',)})
 if passed:
  from datetime import datetime,timezone
  protected=dict(plan['protected'])
  for path in [_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-003.components.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-003.manifest.json'),*(_repo_path(ROOT, 'artifacts/final-model-017')).glob('*')]:
   if path.is_file():protected[str(path.relative_to(ROOT))]=fv.sha(path)
  design={'status':'frozen_before_full500_training_and_test_access','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'candidate':'007','seeds':list(s.SEEDS),'seed_selection':'Allthreefixedseeds,noomissionorreweighting','head_config':h.CONFIG,'effective_auxiliary_weight':0.,'unused_classifier':'Retainedforidenticalinitialization;nottrainedorusedinprediction.','representation':'UnchangedofficialfrozenDINOv2ViTS14full504,normalizedblocks6/9/12CLS+patchmean+populationstd,3456features;verified045train/testcachesreused.','preprocessing':'StandardScalerXandtargetmean/std fittrain500only;all3modelsuseidenticaltrain-onlytransform. NoPCA or testfit.','training':'Threefreshheads100fixedepochs,AdamW/cosine/dropout/regulationunchanged051;seedbase(nofoldoffset)forfull500. Noearlystop orfinalmodelselection.','formula':'.75unchanged003+.25mean3MLP = .375baseline+.1875CNN+.1875compactKRR+sum3MLP/12','weights':{'candidate003':.75,'each_of_3_MLP':1/12},'validation_report':'reports/seed-052-validation.json','validation_report_sha256':fv.sha(_repo_path(ROOT, 'reports/seed-052-validation.json')),'development_rmse':{k:v['mean3_blend_rmse'] for k,v in results.items()},'gain_vs006':{k:v['comparison']['gain'] for k,v in results.items()},'limits':'Repeatedselectionon400;strongtrainingfit,conditionaldiagnosticsnotunseenvalidation;noguaranteedPublicgain.','head_source_sha256':fv.sha(_repo_path(ROOT, 'auxiliary_head_051.py')),'scope':'All500labelsusedonlyfinalfit;formersholdoutnotseparatelyscored.Test1000predictionstabilityonly;nothreshold/scaler/PCA/headfitontest.No upload.','protected':protected}
  fv.write(s.OUT/'candidate-design.json',design)
 print(json.dumps({'valid':True,'gate':passed,'model_count':nmodels,'reload_error':maxerr,'replays':replays,'scores':{k:v['mean3_blend_rmse'] for k,v in results.items()},'seconds':validation['seconds'],'design_frozen':passed}),flush=True)
if __name__=='__main__':main()
