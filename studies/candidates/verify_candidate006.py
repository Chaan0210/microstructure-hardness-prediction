"""Read-only numerical verification of candidate006; writes a new audit report."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np,torch,joblib
import candidate006_pipeline as p
import candidate_pipeline as cp
import dino_frozen_037 as df
import dino_spatial_039 as ds
import dino_multilayer_044 as multi
from dino_regression_037 import FrozenRegressor

def read(path):
 with path.open(newline='') as f:
  reader=csv.DictReader(f);rows=list(reader);return reader.fieldnames,rows

def main():
 torch.set_num_threads(2);d=p.OUT;csvpath=p.CSV;manifest=json.loads(csvpath.with_suffix('.manifest.json').read_text());fit=json.loads((d/'report.json').read_text());decision=json.loads((d/'decision.json').read_text())
 assert p.fv.sha(csvpath)==manifest['submission_sha256'];assert p.fv.sha(d/'report.json')==manifest['model_report_sha256'];assert p.fv.sha(d/'dino-regressor.joblib')==manifest['model_sha256'];assert p.fv.sha(d/'decision.json')==manifest['decision_sha256'];assert p.fv.sha(p.DESIGN)==manifest['design_sha256'];p.check(manifest['protected']);p.check(manifest['source_sha256']);assert manifest['test_fit'] is False and manifest['clipping'] is False
 assert p.fv.sha(csvpath.with_suffix('.components.csv'))==manifest['components_sha256'];assert p.fv.sha(d/'test_features.npz')==manifest['test_feature_sha256'];assert p.fv.sha(d/'train_features.npz')==fit['feature_sha256']
 assert p.fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-005.csv'))==decision['design']['best005_sha256']=='473ef271d35b2c18185690471e2ffbada07a01cf31d07be4c2d489bd76e67ff4';p.check(json.loads((multi.OUT/'plan.json').read_text())['protected'])
 assert p.fv.sha(_repo_path(ROOT, 'data/train.csv'))==fit['train_csv_sha256'];assert p.fv.sha(_repo_path(ROOT, 'data/sample_submission.csv'))==manifest['sample_submission_sha256']
 smoke=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
 for n,h in smoke['official_source_sha256'].items():assert p.fv.sha(df.SOURCE/n)==h
 for n,h in smoke['metadata_sha256'].items():assert p.fv.sha(df.SOURCE/n)==h
 assert p.fv.sha(df.VENDOR/'dinov2_vits14_pretrain.pth')==smoke['source']['dinov2_vits14_pretrain.pth']['sha256']
 fields,rows=read(csvpath);sf,sample=read(_repo_path(ROOT, 'data/sample_submission.csv'));_,comps=read(csvpath.with_suffix('.components.csv'));_,oldrows=read(_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'))
 assert fields==sf==['ID','hardness'] and len(rows)==1000;ids=[r['ID'] for r in rows];assert len(set(ids))==1000 and ids==[r['ID'] for r in sample]==[r['ID'] for r in comps]==[r['ID'] for r in oldrows]
 values=np.array([float(r['hardness']) for r in rows]);assert values.shape==(1000,) and np.isfinite(values).all();components={n:np.array([float(r[n]) for r in comps]) for n in comps[0] if n!='ID'}
 for a in components.values():assert a.shape==(1000,) and np.isfinite(a).all()
 np.testing.assert_allclose(values,.75*components['candidate003']+.25*components['dino_multi'],rtol=0,atol=1e-10);np.testing.assert_array_equal(components['candidate003'],[float(r['hardness']) for r in oldrows]);np.testing.assert_allclose(components['candidate003'],.5*components['baseline']+.25*components['cnn']+.25*components['kernel'],rtol=0,atol=1e-10)
 bundle=joblib.load(_repo_io(d/'dino-regressor.joblib'));model=bundle['model'];train_ids,y=cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);assert bundle['train_ids']==fit['train_ids']==train_ids and model.scaler.n_samples_seen_==500;assert bundle['design_sha256']==manifest['design_sha256'];assert set(fit['training_image_sha256'])==set(train_ids) and len(fit['training_image_sha256'])==500;assert set(manifest['test_image_sha256'])==set(ids) and len(manifest['test_image_sha256'])==1000
 assert fit['inner_selection']['selected']==min(fit['inner_selection']['candidates'],key=lambda a:a['inner_selection_rmse']);assert fit['training_label_range']==[float(y.min()),float(y.max())]
 with np.load(_repo_io(d/'train_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==train_ids;XT=z['X'];assert XT.dtype==np.float64 and XT.shape==(500,3456)
 with np.load(_repo_io(d/'test_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'];assert X.dtype==np.float64 and X.shape==(1000,3456)
 np.testing.assert_allclose(model.scaler.mean_,XT.mean(0),rtol=0,atol=1e-10);np.testing.assert_allclose(model.scaler.var_,XT.var(0),rtol=0,atol=1e-10);assert model.ymean==float(y.mean()) and model.ystd==float(y.std());assert model.alpha==fit['inner_selection']['selected']['alpha']==fit['inner_selection']['selected']['lambda']*3456
 allindices=set(range(500));sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());groups=np.array([sp['group_by_id'][i] for i in train_ids]);seen=[]
 for tr,va in fit['inner_selection']['splits']:assert not set(tr)&set(va) and set(tr)|set(va)==allindices and not set(groups[tr])&set(groups[va]);seen+=va
 assert sorted(seen)==list(range(500));dp=model.predict(X);np.testing.assert_allclose(dp,components['dino_multi'],rtol=0,atol=1e-10)
 for ident,h in fit['training_image_sha256'].items():assert p.fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==h
 for ident,h in manifest['test_image_sha256'].items():assert p.fv.sha(_repo_path(ROOT, 'data/test', f'{ident}.png'))==h
 oldmanifest=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-003.manifest.json')).read_text());folder=_repo_path(ROOT, 'artifacts/final-model-017')
 for n,h in oldmanifest['model_sha256'].items():assert p.fv.sha(folder/n)==h
 assert cp.sources('run-012')==oldmanifest['source_sha256'];baseline=joblib.load(_repo_io(folder/'baseline.joblib'))
 with np.load(_repo_path(ROOT, 'artifacts/test-prediction-features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;XB=z['X']
 bp=sum(baseline['weights'][n]*reg.predict(XB) for n,reg in baseline['models'].items());np.testing.assert_allclose(bp,components['baseline'],rtol=0,atol=1e-10)
 for n,reg in baseline['models'].items():np.testing.assert_allclose(reg.predict(XB),components['baseline_'+n],rtol=0,atol=1e-10)
 # Fixed evenlyspaced10rows chosen without viewing predictions; end-to-end CPU replay.
 ix=np.linspace(0,999,10,dtype=int);chosen=[ids[j] for j in ix];dm=df.model();cpu_features=[]
 sys.path.insert(0,str(_repo_path(ROOT, 'src')))
 from microstructure_v2.features import extract as extract_baseline,read_image
 baseline_source=json.loads((_repo_path(ROOT, 'artifacts/run-005/report.json')).read_text())['source_sha256'];assert baseline_source=={q.name:p.fv.sha(q) for q in (_repo_path(ROOT, 'src/microstructure_v2')).glob('*.py')}
 rawbase=np.stack([extract_baseline(read_image(_repo_path(ROOT, 'data/test', f'{ident}.png')))[0] for ident in chosen]);np.testing.assert_allclose(rawbase,XB[ix],atol=1e-10,rtol=0);rawbp=sum(baseline['weights'][n]*reg.predict(rawbase) for n,reg in baseline['models'].items());np.testing.assert_allclose(rawbp,bp[ix],atol=1e-10,rtol=0)
 for ident in chosen:cpu_features.append(multi.feature(dm,torch.from_numpy(p.input_image(_repo_path(ROOT, 'data/test', f'{ident}.png'))[None])).numpy().reshape(3456))
 cpu_features=np.stack(cpu_features).astype(np.float64);np.testing.assert_allclose(cpu_features,X[ix],rtol=2e-3,atol=2e-3);cpu_dino=model.predict(cpu_features);np.testing.assert_allclose(cpu_dino,dp[ix],rtol=0,atol=1e-3)
 records=[cp.extract_one(i,'test','run-012') for i in chosen];K=np.stack([a[0] for a in records]);imgs=np.stack([a[2] for a in records])[:,None];kernel=joblib.load(_repo_io(folder/'kernel.joblib'));assert kernel['train_ids']==train_ids;kp=kernel['model'].predict(K);np.testing.assert_allclose(kp,components['kernel'][ix],rtol=0,atol=1e-9)
 cc=cp.predict_cnn(imgs,folder/'cnn.pt','cpu');saved=torch.load(_repo_io(folder/'cnn.pt'),map_location='cpu',weights_only=True);assert saved['train_ids']==train_ids;np.testing.assert_allclose(cc,components['cnn'][ix],rtol=0,atol=1e-3)
 reconstructed=.75*(.5*rawbp+.25*kp+.25*cc)+.25*cpu_dino;np.testing.assert_allclose(reconstructed,values[ix],rtol=0,atol=1e-3)
 lo,hi=fit['training_label_range']
 for n,a in components.items():assert p.summary(a,lo,hi)==manifest['component_summary'][n]
 report={'valid':True,'submission_path':str(csvpath),'submission_sha256':p.fv.sha(csvpath),'rows':1000,'columns':fields,'exact_sample_order':True,'duplicate_ids':0,'nan_inf_count':0,'final_prediction_range':[float(values.min()),float(values.max())],'training_label_range':[lo,hi],'component_summary':manifest['component_summary'],'full_cached_feature_regressor_reload_max_abs_error':float(abs(dp-components['dino_multi']).max()),'raw_image_baseline_feature_error':float(abs(rawbase-XB[ix]).max()),'raw_image_baseline_prediction_error':float(abs(rawbp-bp[ix]).max()),'components_and_train_test_cache_hashes_verified':True,'full_baseline_model_reload_max_abs_error':float(abs(bp-components['baseline']).max()),'cpu_dino_feature_reload_max_abs_error':float(abs(cpu_features-X[ix]).max()),'cpu_dino_prediction_reload_max_abs_error':float(abs(cpu_dino-dp[ix]).max()),'cpu_kernel_reload_max_abs_error':float(abs(kp-components['kernel'][ix]).max()),'cpu_cnn_reload_max_abs_error':float(abs(cc-components['cnn'][ix]).max()),'end_to_end_cpu_ensemble_max_abs_error':float(abs(reconstructed-values[ix]).max()),'end_to_end_tolerance':1e-3,'end_to_end_ids':chosen,'train_only_scaler_target_and_inner_fit_verified':True,'frozen_official_source_weights_and_all_images_verified':True,'original003_best005_and043_044_unchanged':True,'source_sha256':p.fv.sha(Path(__file__)),'test_usage':'Prediction/integrity/stabilityonly,nofit/adaptation/clipping','holdout_usage':'Includedinfinal500training,no separateholdoutscore','uploaded':False}
 path=_repo_path(ROOT, 'reports/candidate006-verification.json')
 if path.exists():assert json.loads(path.read_text())==report
 else:path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
