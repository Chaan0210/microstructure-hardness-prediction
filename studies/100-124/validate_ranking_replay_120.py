"""Preserve failed exact-MPS replay; resolve with deterministic CPU intervention and unchanged-label MPS repeat."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,inspect,types
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import ranking_objective_120 as o
import numpy as np,torch
e=o.e

def compare(left,right,exact=False):
 a=e.read(left/'report.json');b=e.read(right/'report.json');assert a['chosen_weight']==b['chosen_weight'];stateerr=0.;prederr=0.;equal=0;stats=True
 for name in a['model_reports_sha256']:
  lp=(left/name).parent;rp=(right/name).parent;x=e.read(lp/'report.json');y=e.read(rp/'report.json')
  for key in ['initial_sha256','order_sha256','dropout_sha256','optimizer_steps','example_visits','eligible_pair_uses']:assert x[key]==y[key],key
  u=torch.load(_repo_io(lp/'model.pt'),map_location='cpu',weights_only=True);v=torch.load(_repo_io(rp/'model.pt'),map_location='cpu',weights_only=True)
  for key in ['mean','scale','var']:assert torch.equal(u[key],v[key])
  for key in ['target_mean','target_std','train_indices','validation_indices','seed','rank_weight']:assert u[key]==v[key]
  err=max(float((u['state'][k]-v['state'][k]).abs().max()) for k in u['state']);stateerr=max(stateerr,err);equal+=err==0.
  with np.load(_repo_io(lp/'predictions.npz')) as z,np.load(_repo_io(rp/'predictions.npz')) as zz:pe=float(abs(z['predictions']-zz['predictions']).max());prederr=max(prederr,pe)
 if exact:assert stateerr==prederr==0. and a['inner_MSE']==b['inner_MSE']
 else:assert stateerr<1e-6 and prederr<1e-3
 return {'models':8,'state_max_error':stateerr,'prediction_max_error':prederr,'bit_exact_models':equal,'initialization_batches_updates_paircounts_stats_exact':True,'chosen_weight_equal':True,'inner_MSE_max_error':max(abs(a['inner_MSE'][k]-b['inner_MSE'][k]) for k in a['inner_MSE'])}

def main():
 o.check();torch.set_num_threads(2);d=o.load('original');base=o.OUT/'original/fold-0';failed=o.OUT/'outer-label-replay';obs=compare(base,failed)
 source=inspect.getsource(o.fit);cpu_source=source.replace("'mps'","'cpu'");assert source.count("'mps'")==3
 plan={'reason':'OriginalexactMPSlabel-replayassertionfailedonlyrankmodels. Preservenativefailureandalloriginalmodels. TestunchangedlabelMPSrepeatandexactCPUcounterfactualreplay; do notmodifyobjective,weights,selections,predictions orpromotioncriterion.','observed_failure':obs,'CPU_method':'Executeunchangedfitfunctionwiththreedevice-literalsmps->cpuandpredictiondevicecpu. Samearchitecture/data/math/epochs/RNG/batches/optimizer. Full8fitnestedouterreplayedtwice,plusinnerheldoutlabel2fits. CPUdeterministicalgorithmsenabled.','MPS_repeat':'Oneunchangedlabel8fitouterrepeat; comparetooriginalandperturbed. NumericaltoleranceforMPSstate1e-6,prediction1e-3(theexistingCPU-MPSreloadtolerance); CPUinterventionrequiresbitexact.','scope':'No newcandidateorhyperparameters;developmentonly. Originalfailedlogpreserved.','source_sha256':e.sha(Path(__file__)),'original_validator_sha256':e.sha(_repo_path(ROOT, 'validate_ranking_objective_120.py')),'original_fit_source':source,'CPU_fit_source':cpu_source,'failed_log_sha256':e.sha(_repo_path(ROOT, 'reports/validate_ranking_objective_120-leakage.log'))};e.write(o.OUT/'replay-validation-amendment.json',plan)
 originalfit=o.fit;proxy=types.SimpleNamespace(**vars(e));proxy.small_predict=lambda m,B,ix: e.small_predict(m,B,ix,device='cpu');ns=dict(vars(o));ns['e']=proxy;exec(compile(cpu_source,'<120CPU-device-only-fit>','exec'),ns);o.fit=ns['fit'];torch.use_deterministic_algorithms(True)
 alter=dict(d);alter['y']=d['y'].copy();va=np.flatnonzero(d['fid']==0);alter['y'][va]=10000+np.arange(len(va))
 cp=o.OUT/'cpu-original-replay';cq=o.OUT/'cpu-outer-label-replay';o.outer(d,'original',0,cp);o.outer(alter,'original',0,cq);cpu=compare(cp,cq,True);cpu_mps=compare(base,cp)
 rec=e.read(cp/'report.json');tr=np.array(rec['inner_splits'][0]['train_indices']);iv=np.array(rec['inner_splits'][0]['validation_indices']);alt=dict(d);alt['y']=d['y'].copy();alt['y'][iv]+=10000;_,rr=o.paired(alt,tr,iv,o.SEED+100,o.OUT/'cpu-inner-label-replay')
 for wi in range(2):assert rr[wi]['state_sha256']==e.read(cp/f'inner-0/weight-{wi}/report.json')['state_sha256']
 e.emit({'CPU_full_nested_label_replay_exact':True,'CPU_inner_label_replay_exact':True,'CPU_MPS':cpu_mps})
 o.fit=originalfit;torch.use_deterministic_algorithms(False);mp=o.OUT/'mps-unchanged-label-replay';o.outer(d,'original',0,mp);repeat=compare(base,mp);cross=compare(failed,mp)
 result={'passed':True,'original_exact_MPS_assertion_failed_preserved':True,'outer80_perturbed_full_nested8models_CPU_states_predictions_choices_exact':True,'inner107_perturbed_fixed2candidate_CPU_states_exact':True,'MPS_perturbed_vs_original':obs,'MPS_unchanged_repeat_vs_original':repeat,'MPS_unchanged_vs_perturbed':cross,'CPU_deterministic_replay':cpu,'CPU_vs_original_MPS':cpu_mps,'old013_components_unchanged':True,'interpretation':'MPSrankmodelretraininghasfloatingpointnondeterminismalsoobservedwithunchangedlabels;deterministicCPUtrainingisbitexactunderheldoutlabelintervention. ActualMPSreplayretainschoicesandexacttrainstatisticsandmatcheswithinrecordedtolerance. Advancedindexedpairbackpropisalikelysource,notprovenbykernelprofiling. No bitexactMPSretrainingclaim.','additional_fits_this_amendment':26,'failed_replay_fits':8,'source_sha256':e.sha(Path(__file__))};e.write(o.OUT/'leakage-validation.json',result);o.check();e.emit(result)
if __name__=='__main__':main()
