"""Locked exploratory008: .5*007 + .5*B-primary; full500 fit, inference only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,argparse,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import dinob_complement_064 as study
import dinob_062 as backbone
import candidate006_pipeline as p6
import candidate_pipeline as cp
fv=study.fv;h=study.h;SEEDS=study.SEEDS
OUT=_repo_path(ROOT, 'artifacts/final-model-065');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-008.csv')
SOURCES=('candidate008_pipeline.py','verify_candidate008.py','dinob_complement_064.py','dinob_mlp_063.py','dinob_mlp_head_063.py','auxiliary_head_051.py','dinob_062.py','dino_frozen_037.py','dino_spatial_039.py','candidate006_pipeline.py','candidate_pipeline.py')

def prepare():
 study.check();vpath=_repo_path(ROOT, 'reports/dinob-complement-064-validation.json');v=json.loads(vpath.read_text());assert v['passed'] and v['models_verified']==45 and v['max_reload_error']==0 and len(v['results'])==3;assert all(r['comparison']['gain']>0 for r in v['results'].values());assert all(r['B100epoch_exact'] and r['S100epoch_historical_state_exact'] for r in v['full_replays']);assert v['confirmed_for_final_fit'] is False
 OUT.mkdir(exist_ok=False);p=study.check();protected=dict(p['protected'])
 for name in ('artifacts/submission-candidate-003.csv','artifacts/submission-candidate-003.components.csv','artifacts/submission-candidate-003.manifest.json','reports/candidate007-verification.json','reports/dinob-complement-064-summary.json','artifacts/dinob-complement-064/plan.json',str(vpath.relative_to(ROOT))):protected[name]=fv.sha(_repo_path(ROOT, name))
 for path in (_repo_path(ROOT, 'artifacts/final-model-017')).iterdir():
  if path.is_file():protected[str(path.relative_to(ROOT))]=fv.sha(path)
 decision={'candidate':'008','status':'Exploratory candidate, not demonstrated significant improvement','locked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':{'source_thread_id':'01a0f244-6000-7496-a632-e5f79ee2c02a','instruction':'Parent explicitly authorized exploratory008 from the fixed50:50 mixture, full500 retraining/test1000 inference/verification. No further ratio/model search. Preserve007.','posthoc_change':True,'prior_rule':'064 agent-chosen per-partition >=3/5 improving folds; observed3/5,2/5,2/5.','replacement':'User explicitly waived fold-majority condition AFTER observing064 results; all3 pooledRMSE gains and deterministic reproduction support making a small-gain exploratory artifact. This is a posthoc operational decision, not preregistered statistical confirmation.','previous_records_unchanged':True,'no_significance_claim':True},'formula':'0.5*candidate007 + 0.5*(0.75*candidate003 + 0.25*B_MLP_mean3)','equivalent_weights':{'candidate003':.75,'S_MLP_mean3':.125,'B_MLP_mean3':.125},'seeds':list(SEEDS),'head_config':h.CONFIG,'effective_aux_weight':0.,'epochs':100,'fixed_before_full500':True,'full500_recipe':'Retrain3 S and3 B MLP heads on all500; retain already-full500 unchanged candidate003 baseline/kernel/CNN. Verify retrained S weights/predictions exactly reproduce007. B frozen officialViT-B14 with same504 descriptors. No fit of backbone.','selection':'No hyperparameter/weight/epoch/seed search or checkpoint selection in final fit. All3seeds included per backbone. Final endpoint100.','holdout':'Former100 holdout included in final500 fit only; never separately evaluated or used for selection.','test':'Only inference and integrity/reload/raw-feature checks after lock; no tuning, clipping, fitting, selection or submission.','validation_report':str(vpath.relative_to(ROOT)),'validation_sha256':fv.sha(vpath),'development_results':{s:{'007_rmse':r['scores']['reference007']['rmse'],'mixed50_rmse':r['scores']['mixed50']['rmse'],'comparison':r['comparison']} for s,r in v['results'].items()},'source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in SOURCES},'protected':protected,'weights':json.loads((backbone.VENDOR/'provenance.json').read_text()),'cost':0,'new_packages':0,'uploaded':False,'limitations':v['limits']+' Gains are small and fold-wise consistency weak. All conditional95% bootstrap intervals include zero; Public/private improvement unknown.'};fv.write(_repo_path(OUT, 'decision.json'),decision);print(json.dumps({'locked':True,'candidate':'008','posthoc_waiver_explicit':True,'formula':decision['formula']}),flush=True)

def check():
 study.check();d=json.loads((_repo_path(OUT, 'decision.json')).read_text());assert d['candidate']=='008' and d['seeds']==list(SEEDS) and d['head_config']==h.CONFIG and d['effective_aux_weight']==0.;assert d['equivalent_weights']=={'candidate003':.75,'S_MLP_mean3':.125,'B_MLP_mean3':.125}
 for name,digest in {**d['source_sha256'],**d['protected']}.items():assert fv.sha(_repo_path(ROOT, name))==digest,name
 assert fv.sha(_repo_path(ROOT, d['validation_report']))==d['validation_sha256'];backbone.check();return d

def extract_B(ids,split):
 torch.set_num_threads(2);assert torch.backends.mps.is_available();m=backbone.model().to('mps');before=h.state_hash(m.state_dict());values=[];hashes={};t=time.perf_counter()
 for j,ident in enumerate(ids):
  path=_repo_path(ROOT, 'data', split, f'{ident}.png');x=torch.from_numpy(p6.input_image(path)[None]).to('mps');values.append(backbone.feature(m,x).cpu().numpy()[0]);hashes[ident]=fv.sha(path)
  if (j+1)%100==0:print(json.dumps({'stage':'B_features','split':split,'completed':j+1,'total':len(ids),'seconds':time.perf_counter()-t}),flush=True)
 X=np.stack(values);assert X.shape==(len(ids),6912) and np.isfinite(X).all();assert before==h.state_hash(m.state_dict()) and all(p.grad is None for p in m.parameters());del m;torch.mps.empty_cache();return X,hashes,{'device':'mps','seconds':time.perf_counter()-t,'weights_unchanged':True,'descriptor_width':6912}

def predict_checkpoint(path,X):
 a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);m=h.Head(a['input_dim']);m.load_state_dict(a['model']);return h.predict(m,(X-a['scaler_mean'].numpy())/a['scaler_scale'].numpy(),a['target_mean'],a['target_std'])

def fit():
 d=check();assert not (_repo_path(OUT, 'report.json')).exists();torch.set_num_threads(2);start=time.perf_counter();ids,y=cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);prior=json.loads((_repo_path(ROOT, 'artifacts/final-model-053/report.json')).read_text());assert prior['train_ids']==ids and prior['train_csv_sha256']==fv.sha(_repo_path(ROOT, 'data/train.csv'));cache=_repo_path(ROOT, prior['train_feature_cache']);assert fv.sha(cache)==prior['train_feature_sha256']
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids;S=z['X'].astype(float)
 assert S.shape==(500,3456) and np.isfinite(S).all();B,hashes,timing=extract_B(ids,'train');assert hashes==prior['training_image_sha256'];B=B.astype(float)
 with np.load(_repo_path(ROOT, 'artifacts/dinob-062/development_features.npz'),allow_pickle=False) as z:
  ix=[ids.index(i) for i in z['ids'].tolist()];np.testing.assert_array_equal(B[ix],z['X'])
 with (_repo_path(OUT, 'train_B_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=B)
 records=[];trainpred={}
 for arm,X in [('S',S),('B',B)]:
  sc,ym,ys,edges,bins=h.preprocess(X,y);xt=torch.tensor(sc.transform(X),dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins,dtype=torch.long)
  for seed in SEEDS:
   m,r=h.fit(xt,yt,bt,seed,0.);path=_repo_path(OUT, f'{arm}-mlp-seed-{seed}.pt');a={'model':m.state_dict(),'input_dim':X.shape[1],'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'bin_edges_unused':torch.tensor(edges),'train_ids':ids,'config':h.CONFIG,'seed':seed,'effective_aux_weight':0.,'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json'))};torch.save(a,_repo_io(path));pv=h.predict(m,xt,ym,ys);re=predict_checkpoint(path,X);np.testing.assert_array_equal(pv,re);trainpred[f'{arm}_{seed}']=pv
   if arm=='S':
    oldrec=next(a for a in prior['models'] if a['seed']==seed)
    for key in ('initial_state_sha256','batch_order_sha256','final_state_sha256'):assert r[key]==oldrec[key]
    olda=torch.load(_repo_path(ROOT, 'artifacts/final-model-053', oldrec['checkpoint']),map_location='cpu',weights_only=True);assert all(torch.equal(val,olda['model'][key]) for key,val in m.state_dict().items())
   r.update(arm=arm,seed=seed,checkpoint=path.name,checkpoint_sha256=fv.sha(path),train_reload_max_error=0.);records.append(r);print(json.dumps({'stage':'fit_head','arm':arm,'seed':seed,'seconds':time.perf_counter()-start}),flush=True)
 with (_repo_path(OUT, 'train_head_predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),**trainpred)
 report={'n_train':500,'train_ids':ids,'train_csv_sha256':fv.sha(_repo_path(ROOT, 'data/train.csv')),'training_image_sha256':hashes,'train_S_cache':str(cache.relative_to(ROOT)),'train_S_sha256':fv.sha(cache),'train_B_cache':str((_repo_path(OUT, 'train_B_features.npz')).relative_to(ROOT)),'train_B_sha256':fv.sha(_repo_path(OUT, 'train_B_features.npz')),'train_head_predictions_sha256':fv.sha(_repo_path(OUT, 'train_head_predictions.npz')),'models':records,'training_label_range':[float(y.min()),float(y.max())],'target_mean':float(y.mean()),'target_std':float(y.std()),'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json')),'no_holdout_score':True,'test_used':False,'hyperparameter_selection':False,'S_exact007_state_reproduction':True,'B_development400_cache_exact':True,'retained003':'Already trained on all500 and immutable; unchanged ensemble component, not retrained.','B_extraction':timing,'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};check();fv.write(_repo_path(OUT, 'report.json'),report);print(json.dumps({'fit_complete':True,'trained_heads':6,'seconds':report['seconds'],'S_exact007_state_reproduction':True}),flush=True)

def predict():
 d=check();torch.set_num_threads(2);assert not CSV.exists() and not CSV.with_suffix('.components.csv').exists() and not CSV.with_suffix('.manifest.json').exists();fit=json.loads((_repo_path(OUT, 'report.json')).read_text());assert fv.sha(_repo_path(OUT, 'decision.json'))==fit['decision_sha256'];ids,_=cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);m7=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-007.manifest.json')).read_text());v7=json.loads((_repo_path(ROOT, 'reports/candidate007-verification.json')).read_text());assert v7['valid'] and fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-007.csv'))==v7['submission_sha256']==m7['submission_sha256'];assert m7['sample_submission_sha256']==fv.sha(_repo_path(ROOT, 'data/sample_submission.csv'));cache=_repo_path(ROOT, m7['test_feature_cache']);assert fv.sha(cache)==m7['test_feature_sha256']
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids;S=z['X'].astype(float)
 assert S.shape==(1000,3456) and np.isfinite(S).all();start=time.perf_counter();B,hashes,timing=extract_B(ids,'test');assert hashes==m7['test_image_sha256'];B=B.astype(float)
 with (_repo_path(OUT, 'test_B_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=B)
 with (_repo_path(ROOT, 'artifacts/submission-candidate-007.csv')).open(newline='') as f:old7=list(csv.DictReader(f))
 with (_repo_path(ROOT, 'artifacts/submission-candidate-003.csv')).open(newline='') as f:old3=list(csv.DictReader(f))
 with (_repo_path(ROOT, 'artifacts/submission-candidate-007.components.csv')).open(newline='') as f:oldc=list(csv.DictReader(f))
 assert [r['ID'] for r in old7]==[r['ID'] for r in old3]==[r['ID'] for r in oldc]==ids;assert fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-007.components.csv'))==m7['components_sha256'];p007=np.array([float(r['hardness']) for r in old7]);p003=np.array([float(r['hardness']) for r in old3]);C={name:np.array([float(r[name]) for r in oldc]) for name in oldc[0] if name in ('baseline','kernel','cnn') or name.startswith('baseline_')};C.update(candidate003=p003,candidate007=p007)
 for rec in fit['models']:
  path=_repo_path(OUT, rec['checkpoint']);assert fv.sha(path)==rec['checkpoint_sha256'];X=S if rec['arm']=='S' else B;C[f"{rec['arm']}_seed_{rec['seed']}"]=predict_checkpoint(path,X)
 for seed in SEEDS:np.testing.assert_array_equal(C[f'S_seed_{seed}'],[float(r[f'mlp_seed_{seed}']) for r in oldc])
 Smean=np.mean([C[f'S_seed_{s}'] for s in SEEDS],axis=0);Bmean=np.mean([C[f'B_seed_{s}'] for s in SEEDS],axis=0);np.testing.assert_array_equal(.75*p003+.25*Smean,p007);Bprimary=.75*p003+.25*Bmean;pred=.5*p007+.5*Bprimary;np.testing.assert_allclose(pred,.75*p003+.125*Smean+.125*Bmean,atol=1e-12,rtol=0);C.update(S_mean3=Smean,B_mean3=Bmean,B_primary=Bprimary,ensemble008=pred);assert len(set(ids))==1000 and all(a.shape==(1000,) and np.isfinite(a).all() for a in C.values());lo,hi=fit['training_label_range'];stats={k:p6.summary(v,lo,hi) for k,v in C.items()};check()
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID',*C]);w.writerows([ident,*[v[j] for v in C.values()]] for j,ident in enumerate(ids))
 with CSV.open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 manifest={'candidate':'008','exploratory':True,'submission_sha256':fv.sha(CSV),'components_sha256':fv.sha(CSV.with_suffix('.components.csv')),'rows':1000,'columns':['ID','hardness'],'exact_sample_order':True,'duplicate_ids':0,'finite':True,'sample_submission_sha256':fv.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'model_directory':str(OUT.relative_to(ROOT)),'model_sha256':{r['checkpoint']:r['checkpoint_sha256'] for r in fit['models']},'model_report_sha256':fv.sha(_repo_path(OUT, 'report.json')),'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json')),'test_S_cache':str(cache.relative_to(ROOT)),'test_S_sha256':fv.sha(cache),'test_B_cache':str((_repo_path(OUT, 'test_B_features.npz')).relative_to(ROOT)),'test_B_sha256':fv.sha(_repo_path(OUT, 'test_B_features.npz')),'test_image_sha256':hashes,'formula':d['formula'],'equivalent_weights':d['equivalent_weights'],'component_summary':stats,'best007_sha256':m7['submission_sha256'],'S_full1000_prediction_exact007':True,'all_seeds_retained':True,'test_fit':False,'clipping':False,'holdout_scored':False,'uploaded':False,'B_extraction':timing,'seconds':time.perf_counter()-start};fv.write(CSV.with_suffix('.manifest.json'),manifest);check();print(json.dumps({'prediction_complete':True,'submission':str(CSV),'sha256':manifest['submission_sha256'],'prediction_range':[float(pred.min()),float(pred.max())],'S_reproduction_exact':True,'seconds':manifest['seconds']}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','fit','predict']);a=ap.parse_args();{'prepare':prepare,'fit':fit,'predict':predict}[a.mode]()
