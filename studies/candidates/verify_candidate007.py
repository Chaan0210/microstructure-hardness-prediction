"""Independent full1000 model replay plus ten raw-DINO checks; no fitting/upload."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(_repo_path(ROOT, 'src')))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
import candidate007_pipeline as p
import candidate_pipeline as cp
import candidate006_pipeline as p6
import auxiliary_head_051 as h
import dino_frozen_037 as df
import dino_multilayer_044 as multi

def read(path):
 with Path(path).open(newline='') as f:r=csv.DictReader(f);return r.fieldnames,list(r)

def main():
 t=time.perf_counter();torch.set_num_threads(2);design=p.gate();manifest=json.loads(p.CSV.with_suffix('.manifest.json').read_text());fit=json.loads((p.OUT/'report.json').read_text());decision=json.loads((p.OUT/'decision.json').read_text());ev=json.loads((_repo_path(ROOT, 'reports/seed-052-validation.json')).read_text());assert set(ev['results'])=={'original','20261031','20261101'};assert len(ev['full_epoch_replays'])==2 and {v['seed'] for v in ev['full_epoch_replays']}=={20261205,20261219};assert ev['model_count']==45
 assert manifest['submission_sha256']==p.fv.sha(p.CSV);assert manifest['components_sha256']==p.fv.sha(p.CSV.with_suffix('.components.csv'));assert manifest['model_report_sha256']==p.fv.sha(p.OUT/'report.json');assert manifest['decision_sha256']==p.fv.sha(p.OUT/'decision.json');assert manifest['design_sha256']==p.fv.sha(p.DESIGN);p.check(decision['source_sha256']);assert manifest['test_fit'] is False and manifest['clipping'] is False and manifest['all_seeds_retained'] is True
 fields,rows=read(p.CSV);sf,sample=read(_repo_path(ROOT, 'data/sample_submission.csv'));_,cr=read(p.CSV.with_suffix('.components.csv'));assert fields==sf==['ID','hardness'];ids=[r['ID'] for r in rows];assert len(ids)==len(set(ids))==1000 and ids==[r['ID'] for r in sample]==[r['ID'] for r in cr];v=np.array([float(r['hardness']) for r in rows]);C={name:np.array([float(r[name]) for r in cr]) for name in cr[0] if name!='ID'};assert np.isfinite(v).all();assert all(np.isfinite(a).all() and a.shape==(1000,) for a in C.values())
 assert p.fv.sha(_repo_path(ROOT, 'data/sample_submission.csv'))==manifest['sample_submission_sha256'];assert p.fv.sha(_repo_path(ROOT, 'data/train.csv'))==fit['train_csv_sha256'];train_ids,y=cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);assert fit['train_ids']==train_ids
 train_cache=_repo_path(ROOT, fit['train_feature_cache']);test_cache=_repo_path(ROOT, manifest['test_feature_cache']);assert p.fv.sha(train_cache)==fit['train_feature_sha256']==manifest['train_feature_sha256'];assert p.fv.sha(test_cache)==manifest['test_feature_sha256']
 with np.load(_repo_io(train_cache),allow_pickle=False) as z:assert z['ids'].tolist()==train_ids;XT=z['X'].astype(float)
 with np.load(_repo_io(test_cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].astype(float)
 assert XT.shape==(500,3456) and X.shape==(1000,3456) and np.isfinite(XT).all() and np.isfinite(X).all();sc=StandardScaler().fit(XT);ym=float(y.mean());ys=float(y.std());assert fit['target_mean']==ym and fit['target_std']==ys;assert fit['training_label_range']==[float(y.min()),float(y.max())]
 assert len(fit['models'])==3 and {r['seed'] for r in fit['models']}==set(p.SEEDS);new=[];models=[];reload_errors=[]
 for seed in p.SEEDS:
  record=next(r for r in fit['models'] if r['seed']==seed);path=p.OUT/record['checkpoint'];assert p.fv.sha(path)==record['checkpoint_sha256']==manifest['model_sha256'][path.name];a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);assert a['seed']==seed and a['aux_weight']==0 and a['config']==h.CONFIG;assert a['train_ids']==train_ids and len(set(a['train_ids']))==500;assert a['design_sha256']==p.fv.sha(p.DESIGN);np.testing.assert_array_equal(a['scaler_mean'].numpy(),sc.mean_);np.testing.assert_array_equal(a['scaler_scale'].numpy(),sc.scale_);assert a['target_mean']==ym and a['target_std']==ys;assert record['curves'][-1]['epoch']==100 and record['classifier_max_parameter_change']==0
  m=h.Head();m.load_state_dict(a['model']);m.eval();models.append(m);pv=h.predict(m,sc.transform(X),ym,ys);err=float(abs(pv-C[f'mlp_seed_{seed}']).max());assert err==0;reload_errors.append(err);new.append(pv)
 new=np.stack(new);mean=new.mean(0);np.testing.assert_array_equal(mean,C['mlp_mean3']);np.testing.assert_array_equal(.75*C['candidate003']+.25*mean,v)
 # Verify allinputbytes against provenance before using them for inference replay.
 assert set(fit['training_image_sha256'])==set(train_ids);assert set(manifest['test_image_sha256'])==set(ids)
 for ident,digest in fit['training_image_sha256'].items():assert p.fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==digest
 for ident,digest in manifest['test_image_sha256'].items():assert p.fv.sha(_repo_path(ROOT, 'data/test', f'{ident}.png'))==digest
 smoke=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
 for name,digest in smoke['official_source_sha256'].items():assert p.fv.sha(df.SOURCE/name)==digest
 for name,digest in smoke['metadata_sha256'].items():assert p.fv.sha(df.SOURCE/name)==digest
 assert p.fv.sha(df.VENDOR/'dinov2_vits14_pretrain.pth')==smoke['source']['dinov2_vits14_pretrain.pth']['sha256']
 oldm=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-003.manifest.json')).read_text());oldfit=json.loads((_repo_path(ROOT, 'artifacts/final-model-017/report.json')).read_text());folder=_repo_path(ROOT, 'artifacts/final-model-017');assert cp.sources('run-012')==oldm['source_sha256'];assert oldfit['training_id_order']==train_ids;assert oldfit['train_csv_sha256']==fit['train_csv_sha256'];assert oldfit['training_image_sha256']==fit['training_image_sha256']
 for name,digest in oldm['model_sha256'].items():assert p.fv.sha(folder/name)==digest
 baseline=joblib.load(_repo_io(folder/'baseline.joblib'));kernel=joblib.load(_repo_io(folder/'kernel.joblib'));assert kernel['train_ids']==train_ids
 cache=_repo_path(ROOT, 'artifacts/test-prediction-features.npz');assert p.fv.sha(cache)==oldm['baseline_test_feature_cache_sha256']
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids and z['hashes'].tolist()==[manifest['test_image_sha256'][i] for i in ids];XB=z['X'];assert z['names'].tolist()==baseline['feature_names']
 basepred=sum(baseline['weights'][name]*reg.predict(XB) for name,reg in baseline['models'].items());np.testing.assert_allclose(basepred,C['baseline'],atol=1e-10,rtol=0)
 for name,reg in baseline['models'].items():np.testing.assert_allclose(reg.predict(XB),C['baseline_'+name],atol=1e-10,rtol=0)
 print(json.dumps({'stage':'all1000_new_heads_and_baseline_reloaded','max_new_error':max(reload_errors),'seconds':time.perf_counter()-t}),flush=True)
 # Recompute unchanged003 compact features and CNN input from ALL1000 PNGs.
 # Thread backend avoids extra interpreter/process caches; no files outsideproject.
 with joblib.parallel_config(backend='threading',n_jobs=2):K,images,imagehash=cp.extract_all(ids,'test','run-012',oldfit['feature_names'])
 assert imagehash==manifest['test_image_sha256'];kp=kernel['model'].predict(K);cnncheckpoint=torch.load(_repo_io(folder/'cnn.pt'),map_location='cpu',weights_only=True);assert cnncheckpoint['train_ids']==train_ids;cc=cp.predict_cnn(images,folder/'cnn.pt','cpu');np.testing.assert_allclose(kp,C['kernel'],atol=1e-9,rtol=0);np.testing.assert_allclose(cc,C['cnn'],atol=1e-3,rtol=0);old=.5*basepred+.25*kp+.25*cc;np.testing.assert_allclose(old,C['candidate003'],atol=1e-3,rtol=0);full=.75*old+.25*mean;np.testing.assert_allclose(full,v,atol=1e-3,rtol=0)
 np.savez_compressed(_repo_io(p.OUT/'verification-test-compact-cnn-inputs.npz'),ids=np.array(ids),compact=K,cnn_images=images)
 print(json.dumps({'stage':'all1000_allregressors_replayed','full_max_error':float(abs(full-v).max()),'seconds':time.perf_counter()-t}),flush=True)
 # Fixed ten evenlyspacedrows: rawPNG -> frozenDINO -> 3MLP -> fullensemble.
 ix=np.linspace(0,999,10,dtype=int);dm=df.model();raw=[]
 for j in ix:raw.append(multi.feature(dm,torch.from_numpy(p6.input_image(_repo_path(ROOT, 'data/test', f'{ids[j]}.png'))[None])).numpy().reshape(3456))
 raw=np.asarray(raw,dtype=float);np.testing.assert_allclose(raw,X[ix],atol=2e-3,rtol=2e-3);rawheads=np.stack([h.predict(m,sc.transform(raw),ym,ys) for m in models]);np.testing.assert_allclose(rawheads,new[:,ix],atol=1e-3,rtol=0)
 from microstructure_v2.features import extract as bextract,read_image
 rawbase=np.stack([bextract(read_image(_repo_path(ROOT, 'data/test', f'{ids[j]}.png')))[0] for j in ix]);np.testing.assert_allclose(rawbase,XB[ix],atol=1e-10,rtol=0);rawbp=sum(baseline['weights'][name]*reg.predict(rawbase) for name,reg in baseline['models'].items());end=.75*(.5*rawbp+.25*kp[ix]+.25*cc[ix])+.25*rawheads.mean(0);np.testing.assert_allclose(end,v[ix],atol=1e-3,rtol=0)
 lo,hi=fit['training_label_range'];stats={name:p6.summary(a,lo,hi) for name,a in C.items()};assert stats==manifest['component_summary'];all_inside=all(a['below_train_min']==0 and a['above_train_max']==0 for a in stats.values());p.gate();p.check(decision['source_sha256'])
 result={'valid':True,'submission_path':str(p.CSV),'submission_sha256':p.fv.sha(p.CSV),'bytes':p.CSV.stat().st_size,'rows':1000,'columns':fields,'exact_sample_order':True,'duplicate_ids':0,'nan_inf':0,'prediction_range':[float(v.min()),float(v.max())],'training_label_range':[lo,hi],'all_component_predictions_inside_training_label_range':all_inside,'component_summary':stats,'full1000_new_MLP_reload_max_error':max(reload_errors),'full1000_baseline_reload_max_error':float(abs(basepred-C['baseline']).max()),'full1000_kernel_reload_max_error':float(abs(kp-C['kernel']).max()),'full1000_CNN_cpu_reload_max_error':float(abs(cc-C['cnn']).max()),'full1000_ensemble_reload_max_error':float(abs(full-v).max()),'raw10_DINO_feature_error':float(abs(raw-X[ix]).max()),'raw10_MLP_prediction_error':float(abs(rawheads-new[:,ix]).max()),'raw10_full_ensemble_error':float(abs(end-v[ix]).max()),'raw10_ids':[ids[j] for j in ix],'raw_replay_tolerance':1e-3,'scaler_target_all500_trainonly_verified':True,'exact3seed_set_and_all3split_gate_verified':True,'all_input_image_hashes_verified':True,'official_source_weights_hashes_verified':True,'best006_and003_unchanged':True,'test_fit':False,'clipping':False,'holdout_scored':False,'uploaded':False,'seconds':time.perf_counter()-t,'design_sha256':p.fv.sha(p.DESIGN),'model_sha256':manifest['model_sha256'],'verification_source_sha256':p.fv.sha(Path(__file__))}
 p.fv.write(_repo_path(ROOT, 'reports/candidate007-verification.json'),result)
 from datetime import datetime,timezone
 handoff={**{k:result[k] for k in ['submission_path','submission_sha256','bytes','rows','columns','exact_sample_order','duplicate_ids','nan_inf','prediction_range','training_label_range','all_component_predictions_inside_training_label_range','full1000_ensemble_reload_max_error','raw10_full_ensemble_error','test_fit','clipping','holdout_scored','uploaded']},'candidate':'007','best006_preserved':True,'seeds':list(p.SEEDS),'fixed_recipe':design['formula'],'design':str(p.DESIGN),'model_directory':str(p.OUT),'verification':'reports/candidate007-verification.json','development_rmse':design['development_rmse'],'gain_vs006':design['gain_vs006'],'limitations':'Repeatedlyreused400;strongtrain-validationgap;~74-78%netSSEgainin10rowsbutgainpositiveaftertheirremoval. NoPublicguarantee.','fit_seconds':fit['seconds'],'prediction_seconds':manifest['seconds'],'verification_seconds':result['seconds'],'model_sha256':manifest['model_sha256'],'cost':0,'running_work':[],'completed_at_utc':datetime.now(timezone.utc).isoformat()};p.fv.write(_repo_path(ROOT, 'reports/candidate007-handoff.json'),handoff)
 text='Candidate007 ready, not uploaded.\nPath: '+str(p.CSV)+'\nSHA256: '+result['submission_sha256']+'\nFormula: '+design['formula']+'\nSeeds: '+str(list(p.SEEDS))+'\n'
 text+='Development RMSE: '+json.dumps(design['development_rmse'])+'\nGains vs006: '+json.dumps(design['gain_vs006'])+'\n'
 text+='1000rows,ID/hardness,exactsampleorder,no duplicates/NaN/Inf. Predictions '+str(result['prediction_range'])+'; allindividualcomponentrangesinside traininglabels '+str([lo,hi])+'. No clipping.\n'
 text+='45developmentmodelreloads and2fulltrainingreplays exact. Full1000ensemble replayerror '+str(result['full1000_ensemble_reload_max_error'])+'; raw10PNG-DINO-MLP-ensembleerror '+str(result['raw10_full_ensemble_error'])+' (tolerance1e-3).\n'
 text+='Full500training only; formerholdout not separately evaluated. Testprediction/integrityonly. Best006/003 unchanged. No cost,installation,upload or runningprocess.\n'
 text+='Limitations: same400developmentexamples repeated; MLP fits training veryclosely. Seedmeans are not always better than everyindividualseed; all3 retained. Influence stresskeepspositivegains but shows concentration in high-benefitrows.\n'
 text+='Reproduction in a project copy retaining verified caches/source/003models: .venv/bin/python -I -B candidate007_pipeline.py fit ; .venv/bin/python -I -B candidate007_pipeline.py predict ; .venv/bin/python -I -B verify_candidate007.py . Outputs exclusive-create; do not overwrite protected originals.\n'
 (_repo_path(ROOT, 'reports/candidate007-results.txt')).write_text(text)
 sp=_repo_path(ROOT, 'reports/status.json');status=json.loads(sp.read_text());status.update(status='candidate007_ready_verified_not_uploaded',latest_completed='reports/candidate007-handoff.json',latest_research='reports/seed-052-summary.json',latest_verification='reports/candidate007-verification.json',new_submission_candidate=str(p.CSV),candidate007_handoff='reports/candidate007-handoff.json',running_work=[],current_study_submission_performed=False,current_study_cost=0,updated_at_utc=datetime.now(timezone.utc).isoformat());sp.write_text(json.dumps(status,indent=2))
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
