"""Record the completed development-only causal sequence without changing results."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
sys.path.insert(0,str(ROOT))
import frozen_validation as fv

def main():
 diag=json.loads((_repo_path(ROOT, 'artifacts/error-diagnostics-068/report.json')).read_text());records=[]
 for name in ('dinob-prenorm-069','dinos-coarse-070','dinob-denoised-071'):
  v=json.loads((_repo_path(ROOT, 'reports', f'{name}-validation.json')).read_text());r=json.loads((_repo_path(ROOT, 'artifacts', name, 'original/report.json')).read_text());p=json.loads((_repo_path(ROOT, 'artifacts', name, 'plan.json')).read_text());assert v['passed'] and not v['eligible_for_candidate'] and r['comparison']['gain']<=0
  for path,digest in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, path))==digest,path
  item={'study':name,'hypothesis':p['hypothesis'],'reference008_rmse':r['scores']['reference008']['rmse'],'primary_rmse':r['scores']['primary']['rmse'],'rmse_gain':r['comparison']['gain'],'positive_folds':r['comparison']['positive_folds'],'full_comparison':r['comparison'],'head_train_rmse':r['mean3_train_rmse_pooled'],'fit_seconds':r['seconds'],'models_verified':v['models_verified'],'full_refits':v['full_replays'],'feature_verification':v['feature_verification'],'decision':'Rejected at prespecified original screen; no resplit/weight sweep/full500/test/candidate/upload.','evidence':{str(q.relative_to(ROOT)):fv.sha(q) for q in [_repo_path(ROOT, 'artifacts', name, 'plan.json'),_repo_path(ROOT, 'artifacts', name, 'original/report.json'),_repo_path(ROOT, 'reports', f'{name}-validation.json')]}}
  records.append(item)
 pub=json.loads((_repo_path(ROOT, 'reports/candidate008-public-submission.json')).read_text());assert fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-008.csv'))==pub['sha256'];assert diag['source_sha256']==fv.sha(_repo_path(ROOT, 'diagnose_errors_068.py'));assert diag['diagnostics_sha256']==fv.sha(_repo_path(ROOT, 'artifacts/error-diagnostics-068/development_diagnostics.npz'))
 summary={'outcome':'Continued beyond diagnosis and each failed study; completed three distinct fixed hypotheses with45OOFheadfits,3benchmarkfits,3fullreplays. No new candidate improves008.',
 'diagnostic_findings':{'input_normalization':'Fixedaffine DINO andglobaltrainingCNN normalization preserve brightness/contrast; perimagebrightnessremoval claimincorrect. CNN128/D4,percentileclipping,tokenLN,andspatialpoolingcanloseinformation inrespectivebranches.','signed_error':'Across20prespecifiedimageproxies,largestmean signed-errorSpearman magnitude~0.051;doesnotjustify simplebrightnessbiascorrection.','absolute_error':'Coarseautocorrelation16 andtilemeanvariation correlations~0.134/~0.131;noiseMAD~0.100. Weakmultipleinspected descriptiveassociations,notcausal evidence.','source':'artifacts/error-diagnostics-068/report.json'},
 'studies':records,
 'learned':[
 '069samewidth/sametraining:removingoutputtokenLayerNorm worsenedall5folds. Retainingrawtokenamplitude didnothelpthisrecipe;notproofallnormalizationisoptimal.',
 '070coarseregionalvariance differsfrom042localvariogramsand067orderlessFisher,butdidnotyieldgeneralizationgain. Spatialinformationaddition aloneisinsufficient;capacityconfoundacknowledged.',
 '071testsinputnoisesuppression separatelyfromtokenamplitudeandspatialpooling;fixedTVdoesnotjustifyadoption basedonoriginalscreen. No strengthretuning.',
 'VerylowheadtrainingRMSE versusOOF indicatesgeneralizationgap. Existing400 repeatedlyreused;all3splitsare not independent. No inferredirreduciblenoisefloor orclaimofattainableperformanceceiling.'
 ],
 'best':{'candidate':'artifacts/submission-candidate-008.csv','sha256':pub['sha256'],'parent_confirmed_public_rmse':11.8898307061,'public_source':'reports/candidate008-public-submission.json','development_rmse':[13.209632611789585,13.180839919990602,13.393004886763789],'protected':True},
 'remaining_problem':'Prizegoalnotachievedorproven. These results eliminate thisdiagnostic-led sequence,notallpossiblemethods. Currentbottleneckisreliableevidenceforanotherdirection andfreshvalidation,notlocalcomputeavailability.',
 'next_useful_evidence':['Officialsyntheticimagegeneration andhardnesslabelconstruction definitions,includingwhichlatentvariablesareactuallyvisible andwhichareunobserved. Thiswouldmakeaphysics-basedresidualhypothesisidentifiable.','Rules-permittedindependentlabeledtraining/validation pairs toassessnewdirections without furtherselectiononthesame400;noholdoutunsealing.','Ifsuchinformationisnotavailable,report008asthecurrentverifiedresultratherthanclaimprize-readiness ormanufactureevidencefromseed/weight sweeps.'],
 'constraints':{'holdout100_separate_access':False,'test_access_during_phase':False,'new_package_or_weights':False,'cost':0,'upload':False,'new_candidate':False,'best008_preserved':True},
 'technical_blockers':[],'running_experiments':[],'source_sha256':fv.sha(Path(__file__)),'diagnostic_report_sha256':fv.sha(_repo_path(ROOT, 'artifacts/error-diagnostics-068/report.json')),'limits':fv.CAVEAT}
 fv.write(_repo_path(ROOT, 'reports/diagnostic-research-068-071-summary.json'),summary);print(json.dumps({'completed':True,'studies':[{k:r[k] for k in ['study','primary_rmse','rmse_gain','positive_folds','models_verified']} for r in records],'best008_sha256':pub['sha256'],'new_candidate':False}),flush=True)
if __name__=='__main__':main()
