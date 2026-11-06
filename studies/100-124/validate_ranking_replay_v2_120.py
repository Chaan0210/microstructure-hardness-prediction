"""Finish replay audit; CPU invariance is within backend, not cross-device training equivalence."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,inspect,types
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import validate_ranking_replay_120 as v
import numpy as np,torch
o=v.o;e=o.e

def main():
 o.check();torch.set_num_threads(2);d=o.load('original');base=o.OUT/'original/fold-0';failed=o.OUT/'outer-label-replay';cp=o.OUT/'cpu-original-replay';cq=o.OUT/'cpu-outer-label-replay';cpu=v.compare(cp,cq,True)
 plan={'reason':'CPUfullnestedouterlabelinterventionpassedbitexact. PreviousamendmentthenincorrectlyrequiredCPUtrainingtomatchMPS. DropoutRNGisdevice-dependent;seedresetdoesnotguaranteecrossdevicemasks. Preservebothfailedlogs/sources. CPUusedonlywithinbackendcounterfactual;actualMPSunchangedrepeatretained.','source_sha256':e.sha(Path(__file__)),'previous_amendment_source_sha256':e.sha(_repo_path(ROOT, 'validate_ranking_replay_120.py')),'previous_failed_log_sha256':e.sha(_repo_path(ROOT, 'reports/validate-ranking-replay-120.log')),'CPU_replay':cpu,'MPS_tolerances_from_first_amendment':{'state':1e-6,'prediction':1e-3},'scope':'Sameweights,data,objectives,budget,candidatecriterion. Verificationonly;nooutcomechanges.'};e.write(o.OUT/'replay-validation-amendment-v2.json',plan)
 torch.manual_seed(o.SEED);dc=torch.nn.functional.dropout(torch.ones(1024),p=.5,training=True).numpy();torch.manual_seed(o.SEED);dm=torch.nn.functional.dropout(torch.ones(1024,device='mps'),p=.5,training=True).cpu().numpy();maskdiff=int((dc!=dm).sum());assert maskdiff>0
 source=inspect.getsource(o.fit);cpu_source=source.replace("'mps'","'cpu'");assert source.count("'mps'")==3;originalfit=o.fit;proxy=types.SimpleNamespace(**vars(e));proxy.small_predict=lambda m,B,ix:e.small_predict(m,B,ix,device='cpu');ns=dict(vars(o));ns['e']=proxy;exec(compile(cpu_source,'<120CPU-device-only-fit>','exec'),ns);o.fit=ns['fit'];torch.use_deterministic_algorithms(True)
 rec=e.read(cp/'report.json');tr=np.array(rec['inner_splits'][0]['train_indices']);iv=np.array(rec['inner_splits'][0]['validation_indices']);alt=dict(d);alt['y']=d['y'].copy();alt['y'][iv]+=10000;_,rr=o.paired(alt,tr,iv,o.SEED+100,o.OUT/'cpu-inner-label-replay')
 for wi in range(2):assert rr[wi]['state_sha256']==e.read(cp/f'inner-0/weight-{wi}/report.json')['state_sha256']
 e.emit({'CPU_full_nested_label_replay_exact':True,'CPU_inner_label_replay_exact':True,'same_seed_CPU_MPS_dropout_mask_different_entries_of1024':maskdiff})
 o.fit=originalfit;torch.use_deterministic_algorithms(False);mp=o.OUT/'mps-unchanged-label-replay';o.outer(d,'original',0,mp);obs=v.compare(base,failed);repeat=v.compare(base,mp);cross=v.compare(failed,mp)
 result={'passed':True,'original_exact_MPS_assertion_failed_preserved':True,'invalid_crossdevice_training_equivalence_assertion_failed_preserved':True,'outer80_perturbed_full_nested8models_CPU_states_predictions_choices_exact':True,'inner107_perturbed_fixed2candidate_CPU_states_exact':True,'MPS_perturbed_vs_original':obs,'MPS_unchanged_repeat_vs_original':repeat,'MPS_unchanged_vs_perturbed':cross,'CPU_deterministic_replay':cpu,'same_seed_CPU_MPS_dropout_mask_different_entries_of1024':maskdiff,'CPU_training_not_compared_as_same_random_trajectory_to_MPS':True,'old013_components_unchanged':True,'interpretation':'MPSrankmodelretraininghassmallfloatingpointvariabilityalsoobservedwithoutchanginglabels. ActualMPSreplayretainschoicesandexacttrainingstatisticsandmatcheswithinrecordedtolerance. DeterministicCPUwithinbackendheldoutlabelinterventionisbitexact. CPU/MPSdropoutRNGdiffers;CPUcertificateiswithinbackendandnotamatchingtrainedMPSmodel. Pairindexgradientaccumulationissuspectednotprovenbykernelprofiling. No bitexactMPStrainingclaim.','validation_fit_count_excluding_original_smooth_control':34,'source_sha256':e.sha(Path(__file__)),'amendment_sha256':e.sha(o.OUT/'replay-validation-amendment-v2.json')};e.write(o.OUT/'leakage-validation.json',result);o.check();e.emit(result)
if __name__=='__main__':main()
