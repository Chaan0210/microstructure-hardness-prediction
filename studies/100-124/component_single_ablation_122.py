"""Two prespecified legacy sub-group removals, reusing the verified121 algorithm."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import component_ablation_121 as a
import numpy as np
oldOUT=a.OUT;e=a.e;n=a.n;r=a.r;OUT=_repo_path(ROOT, 'artifacts/component-single-ablation-122');ARMS=('no_baseline795','no_compact26')
a.MASKS.update(no_baseline795=np.array([0,1,1,1,1],bool),no_compact26=np.array([1,0,1,1,1],bool));a.OUT=OUT;a.ARMS=ARMS

def prepare():
 a.previous.check();old=e.read(oldOUT/'plan.json');pv=e.read(_repo_path(ROOT, 'reports/component-ablation-121-preservation.json'));assert pv['passed'] and e.sha(_repo_path(ROOT, 'reports/component-ablation-121-summary.json'))==pv['summary_sha256'];OUT.mkdir(exist_ok=False)
 protected={**old['protected'],**old['source_sha256'],**{str(p.relative_to(ROOT)):e.sha(p) for p in [oldOUT/'plan.json',oldOUT/'replay-validation.json',_repo_path(ROOT, 'reports/component-ablation-121-summary.json'),_repo_path(ROOT, 'reports/component-ablation-121-preservation.json')]}}
 for split in e.SPLITS:
  assert e.read(oldOUT/f'validation-{split}.json')['passed']
  for p in [oldOUT/split/'report.json',oldOUT/split/'predictions.npz',oldOUT/f'validation-{split}.json']:protected[str(p.relative_to(ROOT))]=e.sha(p)
 plan={'study':'122 twofixedlegacycomponentremovals','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'Parent explicitly asks twofixedcomparisons followedbybestjustifiedconfiguration full500/candidate014; preserve013.','arms':list(ARMS),'method':'Exact121mixtureweights/renormalization/nestedaffine/honestresidual5-wayselection. Onlyremovedmaskchanges. Keep.9affine+.1shortB andphysical57residuals. No newhyperparameters/epochs/features/weightgrid.','candidate_set':['full013','no_legacy_physics',*ARMS],'selection_rule':'Afterindependentvalidation compareall3existingpartitions pooleduntrimmedMSE. Choose lowestamongfull013,121jointremoval,andtwo122singleremovals. Tieswithin1e-10MSEpreferfewerlegacybranches. Requirestrictpooledimprovementover013 for014;noallpartition/fold/CIgate. Allresultsandpaireduncertaintyreported.','uncertainty':'10000IDclusterbootstrapseed20271029,conditionalonmodels;reportpooledandperpartition intervals. Same400historicallyreused;122hypothesisfollowed121results. Notindependentconfirmation andnotselectionadjusted.','scope':'Development400untilselectionlock; no newneuralfit orrawimages required. Finalfitusesall500onlyafterlock;testinferenceandverificationonlyafterfit.','sources_provenance':'Implementationcalls121run/fold;runtimeonlyregisters2newmasks/outputdirectory. 121filesunchanged.','source_sha256':{s:e.sha(_repo_path(ROOT, s)) for s in ['component_single_ablation_122.py','validate_component_single_ablation_122.py','component_ablation_121.py','validate_component_ablation_121.py']},'inputs_sha256':old['inputs_sha256'],'protected':protected,'environment':old['environment'],'prelock_prior_test_artifact_inspection':'Duringimplementationinspection,headersandfirstrowofprevious008/011componentCSVsweredisplayed. No testlabels, fitting, ranking, thresholdorconfigurationdecisionusedthosevalues. No newtestinferencebeforelock;candidatechoiceusesdevelopmentOOFonly.','cost':0,'new_packages':0};e.write(_repo_path(OUT, 'plan.json'),plan);e.emit({'prepared':True,'arms':ARMS,'candidate_set':plan['candidate_set']})
if __name__=='__main__':
 a.torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare',*e.SPLITS]);x=p.parse_args();prepare() if x.mode=='prepare' else a.run(x.mode)
