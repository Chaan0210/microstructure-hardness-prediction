"""Select among exactly four authorized structures by pooled untrimmed development MSE."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import component_single_ablation_122 as b
import numpy as np
a=b.a;e=b.e;NAMES=('full013','no_legacy_physics',*b.ARMS)
def main():
 plan=a.check();P=[];Y=[];F=[];valid={};ids0=None;evidence={}
 for split in e.SPLITS:
  path=b.OUT/split;rec=e.read(path/'report.json');v=e.read(b.OUT/f'validation-{split}.json');assert v['passed'] and v['report_sha256']==e.sha(path/'report.json');valid[split]=v
  with np.load(_repo_io(path/'predictions.npz')) as z,np.load(_repo_io(b.oldOUT/split/'predictions.npz')) as old:
   ids=z['ids'].tolist();assert ids==old['ids'].tolist() and (ids0 is None or ids==ids0);ids0=ids;np.testing.assert_array_equal(z['y'],old['y']);np.testing.assert_array_equal(z['fold'],old['fold']);np.testing.assert_array_equal(z['reference013'],old['reference013']);P.append(np.stack([z['reference013'],old['no_legacy_physics'],*[z[n] for n in b.ARMS]]));F.append(np.stack([z['reference013'],old['frozen_no_legacy_physics'],*[z['frozen_'+n] for n in b.ARMS]]));Y.append(z['y'])
  for q in [path/'predictions.npz',path/'report.json',b.OUT/f'validation-{split}.json']:evidence[str(q.relative_to(ROOT))]=e.sha(q)
 assert e.read(b.OUT/'replay-validation.json')['passed'];P=np.stack(P);Y=np.stack(Y);F=np.stack(F);E=(P-Y[:,None,:])**2;perid=E.mean(0);mse=perid.mean(1);R=np.sqrt(mse);draw=np.random.default_rng(20271029).integers(0,400,(10000,400));boot=np.stack([p[draw].mean(1) for p in perid]);rows=[]
 for j,name in enumerate(NAMES):
  parts=[]
  for s,split in enumerate(e.SPLITS):
   delta=np.sqrt(E[s,j,draw].mean(1))-np.sqrt(E[s,0,draw].mean(1));parts.append({'split':split,'RMSE':float(np.sqrt(E[s,j].mean())),'RMSE_change_vs013':float(np.sqrt(E[s,j].mean())-np.sqrt(E[s,0].mean())),'paired_ID_bootstrap95_RMSE_change_CI':np.quantile(delta,[.025,.975]).tolist()})
  rows.append({'configuration':name,'pooled_MSE':float(mse[j]),'pooled_RMSE':float(R[j]),'RMSE_change_vs013':float(R[j]-R[0]),'ID_cluster95_RMSE_change_CI':np.quantile(np.sqrt(boot[j])-np.sqrt(boot[0]),[.025,.975]).tolist(),'per_partition':parts,'frozen_trainmean_replacement_RMSE':float(np.sqrt(np.mean((F[:,j]-Y)**2)))})
 best=float(mse.min());ties=[j for j,x in enumerate(mse) if x<=best+1e-10];complexity=[2,0,1,1];chosen=min(ties,key=lambda j:(complexity[j],j));eligible=bool(mse[chosen]<mse[0]);mask=a.MASKS['full' if chosen==0 else NAMES[chosen]];w=a.W*mask;w/=w.sum()
 decision={'candidate':'014','eligible':eligible,'selected_configuration':NAMES[chosen],'selection':'MinimumpooleduntrimmedMSEamongexact4authorizedstructures;ties1e-10preferfewerlegacybranches. No allpartition/fold/CIgate.','pooled_MSE':dict(zip(NAMES,mse.tolist())),'pooled_RMSE':dict(zip(NAMES,R.tolist())),'selected_008_component_order':list(a.COLS),'selected_008_normalized_weights':w.tolist(),'formula':'.9*train-only-affine(selected008mixture)+.1*existingBshort+mean3honestselectedphysics57residuals','scope':'Chosenafterseeing121and122onsame400;exploratorypostselectionrecipe,notindependentvalidation. Former100mayenterfinal500fitonly;noholdoutsubgroupscore/testtuning.','authorization':'Parent explicitlyrequestedbestjustifiedconfigurationfull500andcandidate014CSV;retain013andcoordinateuploadwithparent.','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};e.write(b.OUT/'decision.json',decision)
 summary={'study':'122 twolegacy-single-removal followup','status':'candidate014_configuration_locked' if eligible else 'no_improvement','decision':decision,'all_configurations':rows,'validation':{'partitions':valid,'models_reloaded':sum(v['models_reloaded'] for v in valid.values()),'replay':e.read(b.OUT/'replay-validation.json')},'selection_bias':'Same400andknown3partitions;121resultmotivated122andjointremovalwasalreadyknown. Nestedfitsprotectcurrenttraining/selectiononly. Conditionalbootstrapignoresresearchselectionandrefittinguncertainty;noPublic/privategainguarantee.','zeroing_vs_refit':'Frozeninterventionreplacesremovedoutputsbyoutertrainingtargetmeanbutretainsoldcalibratorandoldresidual. Refitproceduredeletesbranchthroughallnestedanchors,renormalizesweights,refitsaffineandresidualtargets/models. Theyareseparateestimands;rawweightzeroing isnotfeatureimportance.','full500_next':'Reuseverifiedremainingbasepredictorstrainedonall500and075/114middle/deepOOF;refitall500calibrator,3anchorcalibrators,and3residualselectors/modelsunderlockedmask. Newneuraltrainingunnecessarybecauseindependentremainingbranchesunchanged. Testonlyafterlockedfit/validation.','scope':{'new_packages':0,'cost':0,'new_submission_CSV_yet':False,'uploaded':False,'prior_test_artifact_inspection':plan['prelock_prior_test_artifact_inspection']},'evidence_sha256':{**evidence,'artifacts/component-single-ablation-122/plan.json':e.sha(b.OUT/'plan.json'),'artifacts/component-single-ablation-122/decision.json':e.sha(b.OUT/'decision.json')},'source_sha256':e.sha(Path(__file__))};e.write(_repo_path(ROOT, 'reports/component-single-ablation-122-summary.json'),summary);a.check();e.emit({'decision':decision,'rows':rows,'summary_sha256':e.sha(_repo_path(ROOT, 'reports/component-single-ablation-122-summary.json'))})
if __name__=='__main__':main()
