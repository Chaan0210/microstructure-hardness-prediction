"""Record fixed B adaptation outcome and protected best without further search."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_adapter_080 as e
import numpy as np
fv=e.fv

def main():
 p=e.check();v=json.loads((_repo_path(ROOT, 'reports/dinob-adapter-080-validation.json')).read_text());assert v['passed'];results={}
 for s,r in v['results'].items():
  out=e.OUT/s;assert fv.sha(out/'report.json')==r['report_sha256'];folds=[json.loads((out/f'fold-{k}/report.json').read_text()) for k in range(5)];report=json.loads((out/'report.json').read_text())
  d=e.inputs(s);null=d['reference009'].copy()
  for k in range(5):
   va=np.flatnonzero(d['fid']==k);fr=json.loads((e.previous.base.OUT/s/f'fold-{k}/report.json').read_text())
   with np.load(_repo_io(out/f'fold-{k}/predictions.npz'),allow_pickle=False) as z:null[va]+=fr['slope']*.125*(z['warm']-z['frozen'])
  nullaudit={'zero_adapter_009_RMSE':fv.rmse(d['y'],null),'zero_adapter_gain_vs_exact009':fv.rmse(d['y'],d['reference009'])-fv.rmse(d['y'],null),'max_prediction_difference':float(abs(null-d['reference009']).max()),'purpose':'Quantify CPU/MPS floating-point control differences separately from learned adapter gain.'}
  results[s]={'zero_adapter_numeric_control':nullaudit,'scores':r['scores'],'comparison':r['comparison'],'train_RMSE_before_pooled':float(np.sqrt(np.mean([f['train_rmse_before']**2 for f in folds]))),'train_RMSE_after_pooled':float(np.sqrt(np.mean([f['train_rmse_after']**2 for f in folds]))),'adapter_up_L2':[f['adapter_up_l2'] for f in folds],'total_folds_seconds':sum(f['seconds'] for f in folds),'max_RSS_bytes':max(f['rss_peak_bytes'] for f in folds),'max_MPS_driver_bytes':max(f['mps_driver_bytes'] for f in folds)}
 original=results['original'];eligible=v['eligible_for_candidate'];pub=json.loads((_repo_path(ROOT, 'reports/candidate009-public-submission.json')).read_text());assert all(fv.sha(_repo_path(ROOT, n))==h for n,h in pub['protected_sha256'].items());r={'study':'080 ViT-B14 finalattention rank4 adapters only','duplicate_audit':p['duplicate_audit'],'config':e.CFG,'control':'Same existing3seedBMLP, foldtrainStandardScaler/targetstats, original pretrainedBweights frozen. All009 affine slopes/centers andweights unchanged; only B representation receives the adapter. No077or079calibration extension.','resources':{'microbenchmark':json.loads((e.OUT/'microbenchmark.json').read_text()),'full_fold_benchmark':json.loads((e.OUT/'benchmark.json').read_text()),'prefix_cache':{k:val for k,val in json.loads((e.OUT/'prefix.json').read_text()).items() if k!='ids'}},'results':results,'verification':{k:val for k,val in v.items() if k not in ['results','limits']},'decision':'Eligible for locked final500 candidate procedure.' if eligible else 'Reject this fixed adapter recipe; preserve best009. No alternate rank, learning rate, epoch, anchor penalty or head setting was tested.','interpretation':'Training fit improvement alone is not evidence of improved generalization. The frozen B heads already closely interpolate training targets; the adapter fits the remaining small training residual under anchoring. This tests one tightly constrained adaptation recipe, not all possible B fine-tuning. Failure does not establish a model capacity ceiling.','next_hypothesis_judgment':'A distinct future question is joint head/adapter training with a matched frozen-backbone head trained under the identical schedule, rather than adapting an already near-interpolating fixed head. It would need a separate preregistered budget and outertrain-only epoch policy; not executed or claimed superior here. Repeating adapter rank/lr/anchor sweeps after this result is not supported by this single-control study.','scope':{'protected009_files_verified':len(pub['protected_sha256']),'best009_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')),'cost':0,'new_install_or_download':False,'holdout_test_selection_access':False,'candidate_created':False,'uploaded':False},'evidence_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/dinob-adapter-080/plan.json','reports/dinob-adapter-080-validation.json']},'source_sha256':fv.sha(Path(__file__)),'limits':p['bias']};fv.write(_repo_path(ROOT, 'reports/dinob-adapter-080-summary.json'),r);print(json.dumps({'eligible':eligible,'results':results,'decision':r['decision'],'scope':r['scope']}),flush=True)
if __name__=='__main__':main()
