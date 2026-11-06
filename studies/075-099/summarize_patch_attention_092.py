"""Report fixed patch aggregation pilot, actual resources and validated decision."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patch_attention_092 as s
import numpy as np
fv=s.fv

def main():
 plan=s.check();v=json.loads((_repo_path(ROOT, 'reports/patch-attention-092-validation.json')).read_text());assert v['passed'];results={}
 for split,r in v['results'].items():
  out=s.OUT/split;assert fv.sha(out/'report.json')==r['report_sha256'];folds=[json.loads((out/f'fold-{k}/report.json').read_text()) for k in range(5)];arms={}
  for arm in s.ARMS:
   a=[f['arms'][arm] for f in folds];ess=[];mx=[]
   for k in range(5):
    with np.load(_repo_io(out/f'fold-{k}'/arm/'predictions.npz'),allow_pickle=False) as z:ess.extend(z['effective_patch_count'].tolist());mx.extend(z['max_weight'].tolist())
   arms[arm]={'parameters_each':12913,'pooled_train_RMSE':float(np.sqrt(np.mean([x['train_RMSE']**2 for x in a]))),'effective_patch_count_quantiles':np.quantile(ess,[0,.1,.25,.5,.75,.9,1]).tolist(),'max_weight_quantiles':np.quantile(mx,[0,.1,.25,.5,.75,.9,1]).tolist(),'initial_loss_mean':float(np.mean([x['curve'][0]['train_loss_online'] for x in a])),'final_loss_mean':float(np.mean([x['curve'][-1]['train_loss_online'] for x in a])),'total_fit_seconds':sum(x['seconds'] for x in a),'max_RSS_bytes':max(x['rss_peak_bytes'] for x in a),'max_MPS_bytes':max(x['mps_driver_bytes'] for x in a)}
  results[split]={**r,'training_and_weight_diagnostics':arms}
 eligible=v['eligible_for_candidate'];pub=s.prior.diag.check_best();r={'study':'092 one fixed local gatedattention pooling pilot with matchedmeancontrol','decision':'Eligible for separate lockedcandidate evaluation.' if eligible else 'No validated materialimprovement. Terminate this patchattention direction as requested; no alternative gate,width,temperature,seed,epoch,blend ormean-onlycandidate. Best011 remains unchanged.','results':results,'config':s.CFG,'architecture':plan['architecture'],'position_policy':plan['position_policy'],'weak_hypothesis':plan['hypothesis'],'screen':plan['screen'],'confirmation':plan['confirmation'],'resource_benchmark':json.loads((s.OUT/'microbenchmark.json').read_text()),'first_full_pair':json.loads((s.OUT/'pilot.json').read_text()),'cache':{k:a for k,a in json.loads((s.OUT/'cache.json').read_text()).items() if k!='ids'},'validation':{k:a for k,a in v.items() if k!='results'},'interpretation':'Matchedparameter comparison isolates learned spatial weighting from modelsize andnonlinear tokenencoding. Gatingfeatures contributevalues inbotharms; no dormantparameter padding. Nonuniform learnedweights indicateanactiveweightingmechanism, not verifiedphysicalimportance. Failure tests thisonecompactmodel andschedule,not allattentionmodels or thedata ceiling.','limits':plan['limits'],'best011':{'CSV':'artifacts/submission-candidate-011.csv','sha256':pub['sha256'],'public_rmse_parent_reported':pub['public_rmse'],'protected_files':len(pub['protected_sha256'])},'scope':{'cost':0,'new_install_or_download':False,'external_data':False,'holdout_test_selection_access':False,'new_candidate_created':False,'uploaded':False,'010_unsubmitted':True},'evidence_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/patch-attention-092/plan.json','reports/patch-attention-092-validation.json','artifacts/patch-attention-092/cache.json']},'source_sha256':fv.sha(Path(__file__))};fv.write(_repo_path(ROOT, 'reports/patch-attention-092-summary.json'),r);print(json.dumps({'eligible':eligible,'decision':r['decision'],'results':results}),flush=True)
if __name__=='__main__':main()
