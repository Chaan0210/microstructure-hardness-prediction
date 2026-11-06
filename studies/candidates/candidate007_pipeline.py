"""Frozen candidate007 final training and prediction; never upload."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import auxiliary_head_051 as h
import candidate_pipeline as cp
import candidate006_pipeline as p6
import frozen_validation as fv
DESIGN=_repo_path(ROOT, 'artifacts/seed-stability-052/candidate-design.json')
OUT=_repo_path(ROOT, 'artifacts/final-model-053')
CSV=_repo_path(ROOT, 'artifacts/submission-candidate-007.csv')
SEEDS=(20261128,20261205,20261219)
SOURCES=('candidate007_pipeline.py','auxiliary_head_051.py','candidate_pipeline.py','candidate006_pipeline.py')

def check(mapping):
 for path,expected in mapping.items():assert fv.sha(_repo_path(ROOT, path))==expected,path

def gate():
 d=json.loads(DESIGN.read_text());vpath=_repo_path(ROOT, d['validation_report']);assert fv.sha(vpath)==d['validation_report_sha256'];v=json.loads(vpath.read_text());assert v['valid'] and v['gate_all_three_improve006'] and v['model_count']==45 and v['max_reload_error']==0;assert all(a['max_parameter_error']==0 for a in v['full_epoch_replays']);assert all(a['comparison']['gain']>0 for a in v['results'].values());assert d['seeds']==list(SEEDS) and d['effective_auxiliary_weight']==0 and d['head_config']==h.CONFIG;assert d['head_source_sha256']==fv.sha(_repo_path(ROOT, 'auxiliary_head_051.py'));check(d['protected']);return d

def model(a):
 m=h.Head();m.load_state_dict(a['model']);return m.eval()

def predict_checkpoint(path,X):
 a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);z=(X-a['scaler_mean'].numpy())/a['scaler_scale'].numpy();return h.predict(model(a),z,a['target_mean'],a['target_std'])

def fit():
 design=gate();OUT.mkdir(exist_ok=False);torch.set_num_threads(2)
 decision={'design_sha256':fv.sha(DESIGN),'design':design,'source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in SOURCES},'validation_sha256':design['validation_report_sha256'],'fixed_before_full500':True,'test_used':False,'fit_scope':'All500forfinalfitonly;formerholdoutnotseparatelyscored;noearlystoporhyperparameterselection','uploaded':False};fv.write(_repo_path(OUT, 'decision.json'),decision)
 # Full500 labels/features are opened only after the validated design is locked.
 ids,y=cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);prior=json.loads((_repo_path(ROOT, 'artifacts/final-model-045/report.json')).read_text());cache=_repo_path(ROOT, 'artifacts/final-model-045/train_features.npz');assert fv.sha(cache)==prior['feature_sha256'];assert prior['train_ids']==ids and prior['train_csv_sha256']==fv.sha(_repo_path(ROOT, 'data/train.csv'))
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].astype(float)
 assert X.shape==(500,3456) and np.isfinite(X).all()
 for ident,digest in prior['training_image_sha256'].items():assert fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==digest
 sc,ym,ys,edges,bins=h.preprocess(X,y);xt=torch.tensor(sc.transform(X),dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins,dtype=torch.long);records=[];start=time.perf_counter()
 for seed in SEEDS:
  m,log=h.fit(xt,yt,bt,seed,0.);path=_repo_path(OUT, f'mlp-seed-{seed}.pt');a={'model':m.state_dict(),'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'target_mean':ym,'target_std':ys,'bin_edges_unused':torch.tensor(edges),'train_ids':ids,'config':h.CONFIG,'seed':seed,'aux_weight':0.,'design_sha256':decision['design_sha256']};torch.save(a,_repo_io(path));before=h.predict(m,xt,ym,ys);after=predict_checkpoint(path,X);error=float(abs(before-after).max());assert error==0;log.update(seed=seed,checkpoint=path.name,checkpoint_sha256=fv.sha(path),train_reload_max_error=error);records.append(log);print(json.dumps({'stage':'fit','seed':seed,'elapsed_seconds':time.perf_counter()-start}),flush=True)
 report={'n_train':500,'train_ids':ids,'train_csv_sha256':fv.sha(_repo_path(ROOT, 'data/train.csv')),'train_feature_cache':str(cache.relative_to(ROOT)),'train_feature_sha256':fv.sha(cache),'training_image_sha256':prior['training_image_sha256'],'training_label_range':[float(y.min()),float(y.max())],'target_mean':ym,'target_std':ys,'models':records,'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json')),'design_sha256':decision['design_sha256'],'no_holdout_score':True,'test_used':False,'hyperparameter_selection':False,'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};check(decision['source_sha256']);gate();fv.write(_repo_path(OUT, 'report.json'),report);print(json.dumps({'fit_complete':True,'seeds':SEEDS,'seconds':report['seconds'],'memory':report['peak_rss_bytes']}),flush=True)

def predict():
 design=gate();torch.set_num_threads(2);report=json.loads((_repo_path(OUT, 'report.json')).read_text());decision=json.loads((_repo_path(OUT, 'decision.json')).read_text());assert report['decision_sha256']==fv.sha(_repo_path(OUT, 'decision.json'));assert decision['design_sha256']==fv.sha(DESIGN);check(decision['source_sha256']);assert not CSV.exists();ids,_=cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False)
 old_manifest=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-003.manifest.json')).read_text());verified6=json.loads((_repo_path(ROOT, 'reports/candidate006-verification.json')).read_text());assert verified6['valid'];manifest6=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-006.manifest.json')).read_text());assert verified6['submission_sha256']==fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-006.csv'))==manifest6['submission_sha256']
 cache=_repo_path(ROOT, 'artifacts/final-model-045/test_features.npz');assert fv.sha(cache)==manifest6['test_feature_sha256'];assert manifest6['sample_submission_sha256']==fv.sha(_repo_path(ROOT, 'data/sample_submission.csv'))
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].astype(float)
 assert X.shape==(1000,3456) and np.isfinite(X).all();assert manifest6['test_image_sha256']==old_manifest['image_sha256']
 for ident,digest in manifest6['test_image_sha256'].items():assert fv.sha(_repo_path(ROOT, 'data/test', f'{ident}.png'))==digest
 with (_repo_path(ROOT, 'artifacts/submission-candidate-003.csv')).open() as f:old=list(csv.DictReader(f))
 with (_repo_path(ROOT, 'artifacts/submission-candidate-003.components.csv')).open() as f:oldc=list(csv.DictReader(f))
 assert [r['ID'] for r in old]==[r['ID'] for r in oldc]==ids;assert fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'))==old_manifest['submission_sha256']
 base=np.array([float(r['hardness']) for r in old]);components={k:np.array([float(r[k]) for r in oldc]) for k in ('baseline','cnn','kernel')};np.testing.assert_allclose(base,.5*components['baseline']+.25*components['cnn']+.25*components['kernel'],atol=1e-10,rtol=0);components['candidate003']=base;start=time.perf_counter()
 for row in report['models']:
  path=_repo_path(OUT, row['checkpoint']);assert fv.sha(path)==row['checkpoint_sha256'];components[f"mlp_seed_{row['seed']}"]=predict_checkpoint(path,X)
 mlps=np.stack([components[f'mlp_seed_{s}'] for s in SEEDS]);mean=mlps.mean(0);pred=.75*base+.25*mean;components.update(mlp_mean3=mean,ensemble007=pred)
 # Retain individualoldbaseline ranges already verified for unchanged models and sameimages.
 with (_repo_path(ROOT, 'artifacts/submission-candidate-006.components.csv')).open() as f:old6=list(csv.DictReader(f))
 assert [r['ID'] for r in old6]==ids;assert fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-006.components.csv'))==manifest6['components_sha256']
 for name in old6[0]:
  if name.startswith('baseline_'):components[name]=np.array([float(r[name]) for r in old6])
 lo,hi=report['training_label_range'];stats={k:p6.summary(v,lo,hi) for k,v in components.items()};assert all(v.shape==(1000,) and np.isfinite(v).all() for v in components.values());assert len(set(ids))==1000
 # Stability diagnostics use training scales only and never alter predictions.
 with np.load(_repo_path(ROOT, report['train_feature_cache']),allow_pickle=False) as z:XT=z['X']
 a=torch.load(_repo_path(OUT, report['models'][0]['checkpoint']),map_location='cpu',weights_only=True);zt=(X-a['scaler_mean'].numpy())/a['scaler_scale'].numpy();ztr=(XT-a['scaler_mean'].numpy())/a['scaler_scale'].numpy();diagnostics={'test_max_abs_train_scaled':float(abs(zt).max()),'train_max_abs_scaled':float(abs(ztr).max()),'test_feature_outside_train_minmax_fraction':float(((X<XT.min(0))|(X>XT.max(0))).mean()),'seed_prediction_std_quantiles':np.percentile(mlps.std(0),[0,50,90,100]).tolist(),'role':'Predictionstabilityonly;no fitting,clipping,thresholdorweightchanges'}
 for suffix in ('.components.csv','.manifest.json'):assert not CSV.with_suffix(suffix).exists()
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID',*components]);w.writerows([ident,*[v[j] for v in components.values()]] for j,ident in enumerate(ids))
 with CSV.open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 gate();check(decision['source_sha256']);manifest={'candidate':'007','submission_sha256':fv.sha(CSV),'components_sha256':fv.sha(CSV.with_suffix('.components.csv')),'sample_submission_sha256':fv.sha(_repo_path(ROOT, 'data/sample_submission.csv')),'rows':1000,'columns':['ID','hardness'],'exact_sample_order':True,'finite':True,'duplicate_ids':0,'model_directory':str(OUT.relative_to(ROOT)),'model_sha256':{r['checkpoint']:r['checkpoint_sha256'] for r in report['models']},'model_report_sha256':fv.sha(_repo_path(OUT, 'report.json')),'decision_sha256':fv.sha(_repo_path(OUT, 'decision.json')),'design_sha256':fv.sha(DESIGN),'test_feature_cache':str(cache.relative_to(ROOT)),'test_feature_sha256':fv.sha(cache),'train_feature_sha256':report['train_feature_sha256'],'test_image_sha256':manifest6['test_image_sha256'],'fixed_weights':design['weights'],'component_summary':stats,'feature_stability':diagnostics,'test_fit':False,'clipping':False,'all_seeds_retained':True,'uploaded':False,'seconds':time.perf_counter()-start};fv.write(CSV.with_suffix('.manifest.json'),manifest);print(json.dumps({'prediction_complete':True,'sha256':manifest['submission_sha256'],'range':[float(pred.min()),float(pred.max())],'component_ranges':{k:[v['min'],v['max']] for k,v in stats.items()},'seconds':manifest['seconds']}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fit','predict']);a=ap.parse_args();fit() if a.mode=='fit' else predict()
