"""Independent008 verification: all1000 reload, all6 fit replay, raw10 descriptors."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(_repo_path(ROOT, 'src')))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import candidate008_pipeline as p
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
import dino_multilayer_044 as multi
fv=p.fv;h=p.h

def read(path):
 with path.open(newline='') as f:r=csv.DictReader(f);return r.fieldnames,list(r)

def main():
 t=time.perf_counter();torch.set_num_threads(2);decision=p.check();fit=json.loads((p.OUT/'report.json').read_text());m=json.loads(p.CSV.with_suffix('.manifest.json').read_text());assert m['submission_sha256']==fv.sha(p.CSV) and m['components_sha256']==fv.sha(p.CSV.with_suffix('.components.csv'));assert m['model_report_sha256']==fv.sha(p.OUT/'report.json') and m['decision_sha256']==fv.sha(p.OUT/'decision.json')==fit['decision_sha256'];assert m['all_seeds_retained'] and not m['test_fit'] and not m['clipping'] and not m['holdout_scored'];assert decision['authorization']['posthoc_change'] and decision['authorization']['no_significance_claim']
 fields,rows=read(p.CSV);sf,sample=read(_repo_path(ROOT, 'data/sample_submission.csv'));cf,cr=read(p.CSV.with_suffix('.components.csv'));ids=[r['ID'] for r in rows];assert fields==sf==['ID','hardness'];assert len(ids)==len(set(ids))==1000 and ids==[r['ID'] for r in sample]==[r['ID'] for r in cr];yhat=np.array([float(r['hardness']) for r in rows]);C={k:np.array([float(r[k]) for r in cr]) for k in cf if k!='ID'};assert np.isfinite(yhat).all() and all(a.shape==(1000,) and np.isfinite(a).all() for a in C.values());assert fv.sha(_repo_path(ROOT, 'data/sample_submission.csv'))==m['sample_submission_sha256'];train_ids,y=p.cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);assert train_ids==fit['train_ids'];assert fv.sha(_repo_path(ROOT, 'data/train.csv'))==fit['train_csv_sha256'];assert fit['no_holdout_score'] and not fit['test_used'] and not fit['hyperparameter_selection'];data={}
 for arm,width in [('S',3456),('B',6912)]:
  for split,metadata in [('train',fit),('test',m)]:
   path=_repo_path(ROOT, metadata[f'{split}_{arm}_cache']);assert fv.sha(path)==metadata[f'{split}_{arm}_sha256']
   with np.load(_repo_io(path),allow_pickle=False) as z:assert z['ids'].tolist()==(train_ids if split=='train' else ids);X=z['X'].astype(float)
   assert X.shape==((500 if split=='train' else 1000),width) and np.isfinite(X).all();data[(arm,split)]=X
 assert fv.sha(p.OUT/'train_head_predictions.npz')==fit['train_head_predictions_sha256']
 with np.load(_repo_io(p.OUT/'train_head_predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==train_ids;trainpred={k:z[k].copy() for k in z.files if k!='ids'}
 assert len(fit['models'])==6 and {(r['arm'],r['seed']) for r in fit['models']}=={(a,s) for a in ('S','B') for s in p.SEEDS};heads={};scalers={};reloaded={};replays=[];ym=float(y.mean());ys=float(y.std());assert fit['target_mean']==ym and fit['target_std']==ys
 for arm,width in [('S',3456),('B',6912)]:
  XT=data[(arm,'train')];X=data[(arm,'test')];sc=StandardScaler().fit(XT);scalers[arm]=sc;edges=np.quantile(y,np.arange(1,7)/7);bins=np.searchsorted(edges,y,side='right');xt=torch.tensor(sc.transform(XT),dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins,dtype=torch.long);heads[arm]=[]
  for seed in p.SEEDS:
   rec=next(r for r in fit['models'] if r['arm']==arm and r['seed']==seed);path=p.OUT/rec['checkpoint'];assert fv.sha(path)==rec['checkpoint_sha256']==m['model_sha256'][path.name];a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);assert a['input_dim']==width and a['seed']==seed and a['train_ids']==train_ids and a['config']==h.CONFIG and a['effective_aux_weight']==0.;assert a['decision_sha256']==fit['decision_sha256'] and a['target_mean']==ym and a['target_std']==ys
   for key,actual in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_),('bin_edges_unused',edges)]:np.testing.assert_array_equal(a[key].numpy(),actual)
   assert h.state_hash(a['model'])==rec['final_state_sha256'];model=h.Head(width);model.load_state_dict(a['model']);heads[arm].append(model);pv=h.predict(model,sc.transform(X),ym,ys);np.testing.assert_array_equal(pv,C[f'{arm}_seed_{seed}']);np.testing.assert_array_equal(h.predict(model,xt,ym,ys),trainpred[f'{arm}_{seed}']);reloaded[f'{arm}_{seed}']=pv
   replay,rr=h.fit(xt,yt,bt,seed,0.)
   for key in ('initial_state_sha256','batch_order_sha256','final_state_sha256'):assert rr[key]==rec[key]
   assert all(torch.equal(val,a['model'][key]) for key,val in replay.state_dict().items());np.testing.assert_array_equal(h.predict(replay,sc.transform(X),ym,ys),pv);replays.append({'arm':arm,'seed':seed,'epochs':100,'parameter_hash_exact':True,'full1000_prediction_error':0.})
   if arm=='S':
    prior=torch.load(_repo_path(ROOT, f'artifacts/final-model-053/mlp-seed-{seed}.pt'),map_location='cpu',weights_only=True);assert all(torch.equal(val,prior['model'][key]) for key,val in a['model'].items())
 print(json.dumps({'stage':'6heads_all500_training_and_all1000_replay_exact','seconds':time.perf_counter()-t}),flush=True)
 Smean=np.mean([reloaded[f'S_{s}'] for s in p.SEEDS],axis=0);Bmean=np.mean([reloaded[f'B_{s}'] for s in p.SEEDS],axis=0);np.testing.assert_array_equal(Smean,C['S_mean3']);np.testing.assert_array_equal(Bmean,C['B_mean3']);p007=.75*C['candidate003']+.25*Smean;Bprimary=.75*C['candidate003']+.25*Bmean;pred=.5*p007+.5*Bprimary;np.testing.assert_array_equal(p007,C['candidate007']);np.testing.assert_array_equal(Bprimary,C['B_primary']);np.testing.assert_array_equal(pred,yhat);np.testing.assert_array_equal(pred,C['ensemble008'])
 _,old7=read(_repo_path(ROOT, 'artifacts/submission-candidate-007.csv'));assert [r['ID'] for r in old7]==ids;np.testing.assert_array_equal(p007,[float(r['hardness']) for r in old7]);_,old3=read(_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'));assert [r['ID'] for r in old3]==ids;np.testing.assert_array_equal(C['candidate003'],[float(r['hardness']) for r in old3]);np.testing.assert_allclose(pred,.75*C['candidate003']+.125*Smean+.125*Bmean,atol=1e-12,rtol=0)
 # Every input image byte is checked; no target information is read from test.
 assert set(fit['training_image_sha256'])==set(train_ids) and set(m['test_image_sha256'])==set(ids)
 for split,mapping in [('train',fit['training_image_sha256']),('test',m['test_image_sha256'])]:
  for ident,digest in mapping.items():assert fv.sha(_repo_path(ROOT, 'data', split, f'{ident}.png'))==digest
 # Reload retained003 components using caches anchored by prior full raw-image validation.
 oldm=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-003.manifest.json')).read_text());oldfit=json.loads((_repo_path(ROOT, 'artifacts/final-model-017/report.json')).read_text());folder=_repo_path(ROOT, 'artifacts/final-model-017');assert oldfit['training_id_order']==train_ids and oldfit['training_image_sha256']==fit['training_image_sha256'];assert p.cp.sources('run-012')==oldm['source_sha256']
 for name,digest in oldm['model_sha256'].items():assert fv.sha(folder/name)==digest
 baseline=joblib.load(_repo_io(folder/'baseline.joblib'));kernel=joblib.load(_repo_io(folder/'kernel.joblib'));assert kernel['train_ids']==train_ids;cache=_repo_path(ROOT, 'artifacts/test-prediction-features.npz');assert fv.sha(cache)==oldm['baseline_test_feature_cache_sha256']
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids and z['hashes'].tolist()==[m['test_image_sha256'][i] for i in ids];XB=z['X'];assert z['names'].tolist()==baseline['feature_names']
 bp=sum(baseline['weights'][name]*reg.predict(XB) for name,reg in baseline['models'].items());np.testing.assert_allclose(bp,C['baseline'],atol=1e-10,rtol=0)
 for name,reg in baseline['models'].items():np.testing.assert_allclose(reg.predict(XB),C['baseline_'+name],atol=1e-10,rtol=0)
 cache=_repo_path(ROOT, 'artifacts/final-model-053/verification-test-compact-cnn-inputs.npz');assert str(cache.relative_to(ROOT)) in decision['protected']
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids;K=z['compact'];images=z['cnn_images']
 kp=kernel['model'].predict(K);cp=p.cp.predict_cnn(images,folder/'cnn.pt','cpu');np.testing.assert_allclose(kp,C['kernel'],atol=1e-9,rtol=0);np.testing.assert_allclose(cp,C['cnn'],atol=1e-3,rtol=0);base003=.5*bp+.25*kp+.25*cp;full=.5*(.75*base003+.25*Smean)+.5*(.75*base003+.25*Bmean);np.testing.assert_allclose(full,yhat,atol=1e-3,rtol=0);fullerror=float(abs(full-yhat).max());print(json.dumps({'stage':'all1000_retained003_models_reloaded','full_ensemble_max_error':fullerror,'seconds':time.perf_counter()-t}),flush=True)
 # Ten fixed evenly-spaced raw PNGs independently re-extract both descriptors on CPU.
 ix=np.linspace(0,999,10,dtype=int);raw_errors={};rawmeans={}
 for arm in ('S','B'):
  dm=p.p6.spatial.base.model() if arm=='S' else p.backbone.model();raw=[]
  for j in ix:
   xb=torch.from_numpy(p.p6.input_image(_repo_path(ROOT, 'data/test', f'{ids[j]}.png'))[None]);v=multi.feature(dm,xb).reshape(1,3456) if arm=='S' else p.backbone.feature(dm,xb);raw.append(v.numpy()[0])
  raw=np.asarray(raw,dtype=float);cached=data[(arm,'test')][ix];np.testing.assert_allclose(raw,cached,atol=2e-3,rtol=2e-3);rp=np.stack([h.predict(model,scalers[arm].transform(raw),ym,ys) for model in heads[arm]]);saved=np.stack([reloaded[f'{arm}_{s}'][ix] for s in p.SEEDS]);np.testing.assert_allclose(rp,saved,atol=1e-3,rtol=0);rawmeans[arm]=rp.mean(0);raw_errors[arm]={'feature_max_error':float(abs(raw-cached).max()),'head_prediction_max_error':float(abs(rp-saved).max())};del dm
 end=.75*base003[ix]+.125*rawmeans['S']+.125*rawmeans['B'];np.testing.assert_allclose(end,yhat[ix],atol=1e-3,rtol=0)
 # Exact CSV replay from six reloaded models and immutable003, independently serialized.
 replay_csv=_repo_path(ROOT, 'artifacts/submission-candidate-008.reproduced.csv')
 with replay_csv.open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 assert fv.sha(replay_csv)==fv.sha(p.CSV);assert replay_csv.read_bytes()==p.CSV.read_bytes();lo,hi=fit['training_label_range'];assert [lo,hi]==[float(y.min()),float(y.max())];stats={k:p.p6.summary(v,lo,hi) for k,v in C.items()};assert stats==m['component_summary'];p.check()
 result={'valid':True,'candidate':'008','exploratory':True,'submission_path':str(p.CSV),'submission_sha256':fv.sha(p.CSV),'bytes':p.CSV.stat().st_size,'rows':1000,'columns':fields,'exact_sample_order':True,'first_id':ids[0],'last_id':ids[-1],'duplicate_ids':0,'missing_or_extra_ids':0,'nan_inf':0,'prediction_range':[float(yhat.min()),float(yhat.max())],'prediction_mean':float(yhat.mean()),'training_label_range':[lo,hi],'negative_predictions':int((yhat<0).sum()),'final_below_train_min':int((yhat<lo).sum()),'final_above_train_max':int((yhat>hi).sum()),'component_summary':stats,'six_head_reload_max_error':0.,'six_full500_100epoch_replays':replays,'all1000_exact_recipe_reload_error':0.,'full1000_all_components_reload_error':fullerror,'CNN_cpu_tolerance':1e-3,'raw10_ids':[ids[j] for j in ix],'raw10_CPU_feature_checks':raw_errors,'raw10_ensemble_error':float(abs(end-yhat[ix]).max()),'raw10_prediction_tolerance':1e-3,'reproduced_csv_path':str(replay_csv),'reproduced_csv_sha256':fv.sha(replay_csv),'reproduced_csv_byte_exact':True,'all500_scalers_target_stats_verified':True,'all1500_image_hashes_verified':True,'best007_preserved':True,'best007_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-007.csv')),'posthoc_operational_gate_waiver':decision['authorization'],'development_results':decision['development_results'],'model_sha256':m['model_sha256'],'decision_sha256':m['decision_sha256'],'model_report_sha256':m['model_report_sha256'],'verification_source_sha256':fv.sha(Path(__file__)),'test_fit':False,'clipping':False,'holdout_scored':False,'uploaded':False,'cost':0,'new_packages':0,'seconds':time.perf_counter()-t,'limitations':decision['limitations']};fv.write(_repo_path(ROOT, 'reports/candidate008-verification.json'),result)
 handoff={k:result[k] for k in ('candidate','exploratory','submission_path','submission_sha256','bytes','rows','columns','exact_sample_order','first_id','last_id','duplicate_ids','missing_or_extra_ids','nan_inf','prediction_range','six_head_reload_max_error','all1000_exact_recipe_reload_error','full1000_all_components_reload_error','raw10_ensemble_error','reproduced_csv_byte_exact','best007_preserved','best007_sha256','development_results','test_fit','clipping','holdout_scored','uploaded','cost','limitations')};handoff.update(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verification_report='reports/candidate008-verification.json',model_directory=str(p.OUT),decision=str(p.OUT/'decision.json'),fixed_formula=decision['formula'],posthoc_gate_change=decision['authorization'],fit_seconds=fit['seconds'],predict_seconds=m['seconds'],verification_seconds=result['seconds'],running_work=[],blockers=[]);fv.write(_repo_path(ROOT, 'reports/candidate008-handoff.json'),handoff)
 print(json.dumps({k:v for k,v in result.items() if k not in ('component_summary','model_sha256','development_results','posthoc_operational_gate_waiver')}),flush=True)
if __name__=='__main__':main()
