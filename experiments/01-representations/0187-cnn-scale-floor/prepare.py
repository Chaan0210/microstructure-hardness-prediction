from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,datetime,shutil
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];D=_repo_path(P, 'experiments/cnn-optimizer-186')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
old=json.loads((_repo_path(D, 'plan.json')).read_text());protected=json.loads((_repo_path(D, 'preservation.json')).read_text())
for p in D.rglob('*'):
 if p.is_file() and not any(v in p.parts for v in ['__pycache__','cache','tmp']):protected[str(p.relative_to(P))]=sha(p)
for n,h in protected.items():assert sha(_repo_path(P, n))==h,n
p=dict(old);p.update(study=187,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='Parentexplicitlyrequests187singletrain-onlyRMSscalefloorcontrol, same186protocol, problem-firstgate, conditionalfullnestedrerun, noholdout/test/download/install/cost,6GB, preserveold.',control186_plan_sha256=sha(_repo_path(D, 'plan.json')),single_factor='Onlyfeature-scale rule. Mean,targetnormalization,backbone,input224wholeimage,head64GELU/dropout.25,100epochwarmhead,12epochadaptation,4optimizerLRrecipes,WD/scheduler/clip/seed/folds/updatesallunchanged.',scaler_policy={'formula':'mu_j=mean_T X_j; var_j=mean_T (X_j-mu_j)^2; s_RMS=sqrt(mean_j var_j); scale_j=max(sqrt(var_j),s_RMS); Z_j=(X_j-mu_j)/scale_j; if allvariancezero:unit scales.','dimensionless_principle':'sigma_j/s_RMS flooredat1. No adjustablecoefficient,epsilonsearch,featuredeletionorpredictionclipping. InvarianttocommonpositivechangeoffeatureunitswhenRMS>0.','fit_scope':'Everyinnertrainandoutertrain independently; fixedscalerduringadaptation; neverfitonvalidationfeatures/labels.','historical_source':'reports/persistence-repair-109-summary.json','historical_sha256':sha(_repo_path(P, 'reports/persistence-repair-109-summary.json'))},warmstart='Fresh100epochfoldtrainheadwithRMSscaler, sameinitialweightsseed/architecture/order/loss/optimizeras186; trained186headsNOTreusedforrepairedcondition. Cacheonlyequivalentfrozenimagefeatures/prefixand186unmodifiedcontrolresults. Pilotcandidatefitsreusedinfullnestedwhenexactfoldrecipe/seedmatch.',selection='Same1864recipesand3innerGroupKFoldmemberships. MinpooledinnerMSEselectsrecipe, samefamilydiagnostics. Choicesmaychangebecauseofscalefloor; thatispartofevaluatedprocedure. Nooutercheckpoint/hyperparameterselection. FreshouterRMSheadthenfamilyrefits; selectedprimaryreusesinnerchosenfamily.',pilot_cases=[{'model':'resnet50','outer':1,'inner':2,'recipes':['adamw_low','adamw_high']},{'model':'resnet50','outer':4,'inner':0,'recipes':['adamw_high']},{'model':'efficientnet_b0','outer':1,'inner':2,'recipes':['adamw_low']}],pilot_gate='Firstreproduce3original186strictCPUfailuresbyfreshsavedmodelCPU/MPSinference. Repairedpre/postadaptationfeatures,predictions,gradientsfinite; CPU/MPSmaxabs<=.002 forALL4pilotfitsandtheirfreshwarmheads. Trainingstd/scalefloor<=1; trainonly/global-units invarianceunitchecks. Knowntriggerfrozenzmaxmustdecrease. Reportprediction/feature rangeswithoutlabel-basedboundsorclipping. No accuracyimprovementrequiredforstabilitygate. Onlythenfullnested.',audit='AllnewmodelsCPUreloadatunchangedabs.002, MPSreloadexact, train-onlyRMSstats/BN/ID/order/actualLR/updateverification. Outerlabelpoisonselectionandinner/outerfulltrainingpoisonreplayssame186. Originalcontrolfailuresremainrecorded; neveroverwritethem.',evaluation='Separate stabilityrecovery fromaccuracy. Sameoriginal4005outerfolds and.9best021+.1new fixedblend. Compare187selected/familyOOFagainst186cachedmatchedproceduresand021. Noresplits/full500/test/newCSV/uploadin187.',fairness=old['fairness']+' BothbackbonestheSAMEtrain-onlyRMSrule, noconditionaloutlier/feature/rowrepair.',scope='Only187outputs; old186and185+best021protected. Noresolution/unfreeze/activation/augmentationchanges ornew tuningaxes.',resources={'CPU_threads':2,'max_RSS_plus_MPS':6000000000,'swap_growth':0,'max_pilot_seconds':900,'max_basic_seconds_per_model':3600,'max_validation_seconds_per_model':900,'CPU_MPS_prediction_absolute_tolerance':.002},no_download=True,cost=0)
for name,obj in [('plan.json',p),('preservation.json',protected)]:
 with (_repo_path(R, name)).open('x') as f:json.dump(obj,f,indent=2)
for n in ['cache','config','tmp']:(_repo_path(R, n)).mkdir()
shutil.copyfile(_repo_path(D, 'common.py'),_repo_path(R, 'common.py'));shutil.copyfile(_repo_path(D, 'launch.py'),_repo_path(R, 'launch.py'))
s=(_repo_path(R, 'launch.py')).read_text().replace("['run.py','validate.py','summarize.py']","['run.py','probe.py','validate.py','summarize.py']");(_repo_path(R, 'launch.py')).write_text(s)
s=(_repo_path(D, 'run.py')).read_text();loc="RECIPES=c.plan['recipes']"
new='''RECIPES=c.plan['recipes']
D=P/'experiments/cnn-optimizer-186'
def floor_stats(X,y,T,V,seed):
 mu=X[T].mean(0);var=X[T].var(0);rms=float(np.sqrt(var.mean()));scale=np.maximum(np.sqrt(var),rms) if rms>0 else np.ones_like(var)
 return {'feature_mean':torch.tensor(mu),'feature_var':torch.tensor(var),'feature_scale':torch.tensor(scale),'feature_RMS_floor':rms,'scaler_policy':'train_RMS_floor','target_mean':float(y[T].mean()),'target_std':float(y[T].std()),'train':T.tolist(),'validation':V.tolist(),'seed':seed,'dim':X.shape[1]}
old.stats=floor_stats
'''
s=s.replace(loc,new)
s=s.replace("global CURRENT_MODEL;CURRENT_MODEL=name;c.check();d=", "global CURRENT_MODEL;CURRENT_MODEL=name;c.check();assert c.read(R/'problem-gate.json')['passed'];c.emit({'started_UTC':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'model':name,'stage':'nested'});d=")
s=s.replace("o=ck(B/name/'original'/f'fold-{k}/frozen100.pt');assert o['train']==T.tolist() and o['validation']==V.tolist()", "o=save_head(path/'initial-head.pt',X,d['y'],T,V,202710185+k);assert o['train']==T.tolist() and o['validation']==V.tolist()")
s=s.replace("name+'-continued-run'", "name+'-nested-run'")
s=s.replace("prefix,tail=c.model(out.parts[-4] if False else CURRENT_MODEL)","prefix,tail=c.model(CURRENT_MODEL)")
(_repo_path(R, 'run.py')).write_text(s)
for n in ['common.py','launch.py','run.py']:compile((_repo_path(R, n)).read_text(),str(_repo_path(R, n)),'exec')
print(json.dumps({'prepared':True,'protected_files':len(protected),'plan_sha256':sha(_repo_path(R, 'plan.json'))}))
