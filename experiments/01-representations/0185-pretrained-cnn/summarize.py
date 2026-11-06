import common as c
from common import R,P
from email.parser import Parser
import time
p=c.check();prov=c.read(R/'provenance.json');results={};eligible=[]
for n in p['models']:
 assert c.read(R/n/'original/validation.json')['passed'];results[n]={}
 for arm in ['frozen100','frozen112','finetune112']:
  z=c.read(R/n/'original'/f'{arm}-report.json');results[n][arm]=z
  if arm!='frozen112' and z['advance_gate']:eligible.append([n,arm])
resources={}
for path in R.glob('*-resources.json'):
 z=c.read(path);trace=z['trace'];resources[path.stem]={'seconds':trace[-1]['seconds'],'peak_RSS':max(x['peak_RSS'] for x in trace),'peak_RSS_plus_MPS':max(x['RSS_plus_MPS'] for x in trace),'max_swap':max(x['swap'] for x in trace),'swap_growth':max(x['swap'] for x in trace)-z['initial_swap']};assert resources[path.stem]['peak_RSS_plus_MPS']<6000000000 and resources[path.stem]['swap_growth']==0
packages=[]
for d in sorted((R/'.venv/lib/python3.13/site-packages').glob('*.dist-info')):
 m=Parser().parsestr((d/'METADATA').read_text());packages.append({'name':m['Name'],'version':m['Version'],'metadata_sha256':c.sha(d/'METADATA')})
c.write(R/'installed-packages.json',packages)
with (R/'requirements.lock').open('x') as f:f.write('\n'.join(x['name']+'=='+x['version'] for x in packages)+'\n')
summary={'study':185,'decision':'original_results_await_resplit' if eligible else 'reject_both_current_recipes_keep_best021','results':results,'eligible_resplits':eligible,'resplits_executed':False,'best021_original_OOF':12.786200775750695,'best021_Public_from_parent':11.4878517800,'best021_CSV_sha256':c.sha(P/'artifacts/submission-candidate-021.csv'),'download_bytes_wheel_and_two_weights':prov['wheel']['bytes']+sum(x['bytes'] for x in prov['weights'].values()),'weights':prov['weights'],'torchvision_wheel':prov['wheel'],'license_record':'licenses/torchvision-BSD-3-Clause.txt; weight-specific/training-data caveat retained in provenance.json; localresearch only. No extra terms,login,paidAPI orredistribution.','environment':'Separateuvvenv withindependentAPFSclonesofexistingdeps; torch2.14.0; torchvision0.29.0installedonlyhere. Originalenvunchanged.','scope':'development400only, original5fold; 2encoders x5fold x3headcheckpoints=30savedmodels; 10actualtailadaptations; noholdout/test/full500/submissionCSV/upload.','validation':{n:c.read(R/n/'original/validation.json') for n in p['models']},'resources':resources,'cost':0,'initial_smoke_failure_and_repair':c.read(R/'smoke-repair.json'),'actual_tuning_this_study':'Frozen100head versusfixed12epochlaststageadaptation andmatched12epochfrozenheadcontrol. Nooptimizer/LR/WD/scheduler/activationsearchperformed.','future_tuning_design_not_run':p['future_tuning_not_executed'],'history_audit':p['history_audit'],'limits':'Singlefixedseed/recipe, full224view instead ofofficialcenter-crop; classificationImageNetCNN versusDINOselfsupervised504 is notpurearchitecturecomparison. Outcomes donotproveallpretrainedCNN/optimizerregimesfail. Same400adaptivelyreused; intervalsnotselectionadjusted.','preservation_verified':True,'protected_file_count':len(c.read(R/'preservation.json')),'source_sha256':{n:c.sha(R/n) for n in ['setup.py','prepare.py','common.py','features.py','train.py','validate.py','summarize.py','launch.py','plan.json']}}
c.write(R/'summary.json',summary)
lines=['Study185: '+summary['decision'],'','model / arm / standalone RMSE / fixed10%blend RMSE']
for n,arms in results.items():
 for arm,z in arms.items():lines.append(f"{n} / {arm} / {z['RMSE']['standalone']:.10f} / {z['RMSE']['blend10']:.10f}")
lines.extend(['','best021 original OOF: 12.7862007758','Frozen/finetune andoptimizer-tuningareseparate. Optimizercomparisonnotrun; boundedfutureplan inplan.json.','No holdout/test, no full500, no newCSV, no upload, cost0.','All30savedmodelsindependentlyCPUreloaded; BN/prefix/trainstats/ID/finite/orderchecks passed.','Originalenvironmentandbestfilesunchanged.','summarySHA256: '+c.sha(R/'summary.json')])
with (R/'RESULTS.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
c.write(P/'reports/pretrained-cnn-185-summary.json',{'summary_path':str((R/'summary.json').relative_to(P)),'summary_sha256':c.sha(R/'summary.json'),'decision':summary['decision'],'results':results,'eligible_resplits':eligible,'best021_preserved':True,'holdout_test_used':False,'optimizer_search_executed':False})
c.emit({'decision':summary['decision'],'eligible_resplits':eligible,'summary_sha256':c.sha(R/'summary.json'),'resources_max_RSS_plus_MPS':max(z['peak_RSS_plus_MPS'] for z in resources.values())})
