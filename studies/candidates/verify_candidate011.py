"""Verify all500 fit/replay, immutable assets and1000 reload/CSV order/finite/hash."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv,time,hashlib,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate011_pipeline as e
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
fv=e.fv

def main():
 decision=e.check();torch.set_num_threads(2);start=time.perf_counter();d=e.training();r=json.loads((e.OUT/'report.json').read_text());rec=json.loads((e.OUT/'head/report.json').read_text());assert fv.sha(e.OUT/'head/report.json')==r['head_report_sha256'];assert fv.sha(e.OUT/'head/model.pt')==r['checkpoint_sha256']==rec['checkpoint_sha256'];assert fv.sha(e.OUT/'head/training_predictions.npz')==rec['predictions_sha256'];a=torch.load(_repo_io(e.OUT/'head/model.pt'),weights_only=True,map_location='cpu');sc=StandardScaler().fit(d['B']);ym=float(d['y'].mean());ys=float(d['y'].std());assert a['train_ids']==d['ids']==r['train_ids'] and a['n_train']==500 and a['config']==e.CFG and a['target_mean']==ym and a['target_std']==ys and len(set(d['ids']))==500
 for n,val in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(a[n].numpy(),val)
 model=e.e.Model(e.e.backbone.model(),sc,e.CFG['seed'],False);assert e.e.h.state_hash(model.state_dict())==rec['initial_state_sha256']==a['initial_state_sha256'];assert e.e.fixed_hash(model)==rec['fixed_state_sha256']==a['fixed_state_sha256'];assert e.e.h.state_hash(a['state'])==rec['final_state_sha256']
 for n,val in model.state_dict().items():
  if not n.startswith('head.'):assert torch.equal(val,a['state'][n])
 orders=np.random.default_rng(e.CFG['seed']).permuted(np.tile(np.arange(500),(12,1)),axis=1);seeds=np.array([e.e.seed_for(0,ep,j) for ep in range(12) for j in range(0,500,2)],dtype=np.int64);assert rec['batch_order_sha256']==hashlib.sha256(orders.tobytes()).hexdigest() and rec['dropout_seeds_sha256']==hashlib.sha256(seeds.tobytes()).hexdigest();assert len(rec['curve'])==12
 for ep,c in enumerate(rec['curve']):assert c['head_lr']==e.CFG['head_min_lr']+.5*(e.CFG['head_lr']-e.CFG['head_min_lr'])*(1+math.cos(math.pi*ep/12))
 tp=e.infer(d['B'],a)
 with np.load(_repo_io(e.OUT/'head/training_predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['predictions'],tp)
 rr=e.train(d,e.OUT/'validation-replay');assert rr['final_state_sha256']==rec['final_state_sha256'];assert rr['initial_state_sha256']==rec['initial_state_sha256'] and rr['batch_order_sha256']==rec['batch_order_sha256'] and rr['dropout_seeds_sha256']==rec['dropout_seeds_sha256']
 with np.load(_repo_io(e.OUT/'validation-replay/training_predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['predictions'],tp)
 ids,B,old,base=e.test_inputs();man=json.loads(e.CSV.with_suffix('.manifest.json').read_text());assert fv.sha(e.CSV)==man['submission_sha256'] and fv.sha(e.CSV.with_suffix('.components.csv'))==man['components_sha256'];assert fv.sha(e.OUT/'test_predictions.npz')==man['predictions_sha256'] and fv.sha(e.OUT/'report.json')==man['model_report_sha256'];assert fv.sha(_repo_path(ROOT, 'data/sample_submission.csv'))==man['sample_submission_sha256'];new=e.infer(B,a);p=.9*old+.1*new
 with np.load(_repo_io(e.OUT/'test_predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['candidate009'],old);np.testing.assert_array_equal(z['frozen_head'],new);np.testing.assert_array_equal(z['primary'],p)
 cp=e.infer(B,a,'cpu');cpuerr=float(abs(cp-new).max());assert cpuerr<.003
 with e.CSV.open(newline='') as f:reader=csv.DictReader(f);assert reader.fieldnames==['ID','hardness'];rows=list(reader)
 with e.CSV.with_suffix('.components.csv').open(newline='') as f:parts=list(csv.DictReader(f))
 assert [x['ID'] for x in rows]==[x['ID'] for x in parts]==ids and len(rows)==1000 and len(set(ids))==1000 and np.isfinite(p).all();np.testing.assert_array_equal(p,[float(x['hardness']) for x in rows]);np.testing.assert_array_equal(old,[float(x['candidate009']) for x in parts]);np.testing.assert_array_equal(new,[float(x['frozen_head']) for x in parts]);np.testing.assert_array_equal(p,[float(x['candidate011']) for x in parts])
 with e.CSV.with_suffix('.reproduced.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,p))
 assert fv.sha(e.CSV.with_suffix('.reproduced.csv'))==man['submission_sha256']
 for ident,digest in r['training_image_sha256'].items():assert fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==digest
 for ident,digest in man['test_image_sha256'].items():assert fv.sha(_repo_path(ROOT, 'data/test', f'{ident}.png'))==digest
 assert fv.sha(_repo_path(ROOT, 'artifacts/final-model-075/calibrator.json'))==man['unchanged009_calibrator_sha256']==r['unchanged009_calibrator_sha256'];e.check();result={'valid':True,'candidate':'011','submission_sha256':fv.sha(e.CSV),'rows':1000,'columns':['ID','hardness'],'exact_sample_order':True,'finite':True,'duplicates':0,'CSV_byte_exact_reproduction':True,'full500_refit_state_exact':True,'full500_train_prediction_reload_error':0.,'full1000_MPS_reload_error':0.,'full1000_CPU_MPS_max_error':cpuerr,'train_only_scaler_targets_verified':True,'backbone_unchanged':True,'all1500_image_hashes_verified':True,'test_features':'Previously validated official frozen B6912 cache; source hash and all1000 image hashes verified. Inference from saved new head repeated on MPS and CPU.','no_new_calibration':True,'formula':decision['formula'],'selection_bias':decision['selection_bias'],'protected009_sha256':man['protected009_sha256'],'protected010_sha256':man['protected010_sha256'],'010_unsubmitted':True,'uploaded':False,'cost':0,'holdout_scored':False,'prediction_range':man['prediction_range'],'difference_from009_RMS':man['difference_from009_RMS'],'source_sha256':fv.sha(Path(__file__)),'seconds':time.perf_counter()-start};fv.write(_repo_path(ROOT, 'reports/candidate011-verification.json'),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
