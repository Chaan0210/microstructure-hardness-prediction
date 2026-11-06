from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json,csv,io,hashlib,datetime
from pathlib import Path
R=_RepoPath(__file__).resolve().parents[3];O=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def J(p):return json.loads(p.read_text())
def W(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
plan=J(_repo_path(O, 'plan.json'));s=J(_repo_path(O, 'summary.json'));v=J(_repo_path(O, 'validation.json'));sv=J(_repo_path(O, 'score-validation.json'));done=J(_repo_path(O, 'study-complete.json'));old=J(_repo_path(O, 'baseline-validation.json'))
assert v['passed'] and sv['passed'] and done['passed'] and done['new_fits']==45
assert sv['summary_sha256']==H(_repo_path(O, 'summary.json'))
for p,h in {**plan['inputs_sha256'],**plan['protected_sha256']}.items():assert H(_repo_path(R, p))==h,p
assert H(_repo_path(R, 'reports/strategy-experiment-register.csv'))==H(_repo_path(O, 'strategy-experiment-register.before.csv')) and H(_repo_path(R, 'README.md'))==H(_repo_path(O, 'README.before.md'))
now=datetime.datetime.now(datetime.timezone.utc).isoformat();strong=s['strong_development_gate']
status='완료·강한 개발단계 기준 통과·best021 보존·최종단계 미실행' if strong else '완료·고정 3→6seed 강한 채택기준 미달·best021 유지'
resources={}
for stage in ['study','validation']:
 trace=J(_repo_path(O, f'{stage}-resources.json'));samples=trace['samples'];resources[stage]={'seconds':samples[-1]['seconds'],'maximum_RSS_plus_MPS_driver_bytes':max(max(x['RSS'],x['peak_RSS'])+x['MPS_driver'] for x in samples),'swap_growth_bytes':max(x['swap'] for x in samples)-trace['initial_swap']}
variance=s['variance'];decrease=100*(1-variance['partition_prediction_variance_primary']/variance['partition_prediction_variance_reference'])
seed_summary={}
for name in ['old3','new3','all6']:
 seed_summary[name]={key:sum(s['seed_variance'][split][name][key] for split in plan['splits'])/3 for key in ['population_seed_variance','sample_seed_variance','conditional_raw_mean_variance_proxy','conditional_final_direct_mean_variance_proxy']}
interpretation={
 'study':203,'UTC':now,'status':status,'actual_execution':'45 new CNN fits:3fixednewseeds x5folds x3existingpartitions;800updates each,36000updates total.45 oldmodelreloads;0old training reruns. All6equallyretained.',
 'duplicate_audit':plan['duplicate_audit'],
 'primary_formula':plan['formula'],'new_seeds':plan['new_seeds'],'old_seeds':plan['old_seeds'],
 'results':{'pooled':s['pooled'],'by_partition':{split:{key:r[key] for key in ['reference_RMSE','primary_RMSE','gain','positive_folds','conditional_CI95']} for split,r in s['by_partition'].items()},'criteria':s['criteria'],'strong_development_gate':strong,'variance':variance,'acrosspartition_variance_reduction_percent':decrease,'seed_summary':seed_summary},
 'assessment_ko':[
  '202의CNN직접분산88%라는진단을근거로현재021의CNN3seed→6seed를실제학습·평가했다. 기존018/021/133의1→3평균을반복하지않았으며새3seed는결과전에고정했다.',
  '후단calibration과residual을고정하고현재021의CNN평균만치환했다. 따라서그학습된후단이기존3seed에의존한다는한계가남는다. 새로운6seed전체pipeline의nested재학습효과로해석하지않는다.',
  '분할간예측분산은동일400이미지의세기존재분할에대한조건부민감도다. seed분산/n값은3개또는6개실현seed에서추정한독립draw가정의proxy이며400이미지가독립seed반복400회가되지는않는다.',
  '개별seed·new3평균은진단전용으로모두기록했다. 좋은seed만남기거나평균비중을바꾸지않았다. 기존및신규6개를항상동일평균했다.',
  '400ID묶음bootstrap은세분할반복측정을함께재표집했다. 고정된모델의조건부CI이며누적선택·학습재추정·알수없는생성family의존성을보정하지않는다.',
  '현재범위에서full500/holdout/test/새CSV/제출은실행하지않는다. 강한개발기준통과여부와별개로기존best021을보존하고결과를 정리한다.'
 ],
 'validation':{'old45_MPS_exact':old['MPS_exact_count']==45,'old_max_CPU_error':old['max_CPU_error'],'new45_MPS_reload_exact':v['new_MPS_reload_exact'],'new_max_CPU_error':v['max_CPU_MPS_error'],'independent_fullensemble_max_error':v['max_independent_fullensemble_error'],'all10000bootstrap_count_replay':True,'bootstrap_max_error':sv['bootstrap_max_abs_error'],'seed_variance_max_error':sv['independent_seed_variance_max_abs_error'],'heldout_labels_masked_in45trainer_calls':True,'updates_initialization_D4_LR_last5averages_verified':True},
 'resources':{**resources,'sum45fit_seconds':done['sum_fit_seconds'],'maximum_monitor_bytes':max(x['maximum_RSS_plus_MPS_driver_bytes'] for x in resources.values()),'CPU_threads':2,'device':'MPS','cost':0,'timing_scope':'Instrumentedexecution;planning/coding/toolapproval/reportingexcluded'},
 'scope':{'new_model_fits':45,'old_model_training_replays':0,'new_packages':0,'downloads':0,'externalAI_calls':0,'network':False,'holdout_test_full500':False,'new_submission_CSV':False,'upload':False,'old_best_preserved':True},
 'failures_preserved':'Before maintraining: -I scriptdir import fix; originalrun009 lacksvalidationIDs andwasverifiedbyexacttrainIDcomplement/cachedOOF. Both preserve originalsources and failuretrace, no seed/recipe/gatechanges. See amendment files.',
 'remaining_blockers':'No execution blocker after successful validation; independentgeneralization andfrozen-downstreamlimitations remain.'}
interpretation['raw_CNN_RMSE']={split:{name:s['seed_variance'][split][name]['raw_mean_RMSE'] for name in ['old3','new3','all6']} for split in plan['splits']}
interpretation['seed_proxy_reduction_percent']=100*(1-seed_summary['all6']['conditional_final_direct_mean_variance_proxy']/seed_summary['old3']['conditional_final_direct_mean_variance_proxy'])
interpretation['assessment_ko'].append('CNN단독 mean6는원본14.31624→14.05384/재분할A13.91865→13.85544/재분할B14.30711→14.24089로모두개선했다. 최종앙상블은두재분할악화여서단독정확도·분산감소만으로최종보완성개선을보증하지않음을확인했다.')
interpretation['assessment_ko'].append('pooled이득.005503의조건부CI는[-.025968,.036307]이다. 상위1수혜가순SSE이득의75.75%이고상위5수혜제외gain은-.011088이다. 동일400재사용하의집중도진단이며표본제거로새후보를만들지않았다.')
interpretation['failures_preserved']+=' Pre-score static review cast NumPy boolean to native bool for JSON; no mathematical/selection change, prior source/lock retained.'
W(_repo_path(O, 'interpretation.json'),interpretation)
report={**s,'status':status,'validation_passed':True,'score_validation_passed':True,'best021_preserved':True,'new_CSV':False,'resources':interpretation['resources'],'seed_summary':seed_summary,'evidence':'experiments/cnn-seed-expand-203/','interpretation_sha256':H(_repo_path(O, 'interpretation.json'))}
W(_repo_path(R, 'reports/cnn-seed-expand-203-summary.json'),report)
registry=_repo_path(R, 'reports/strategy-experiment-register.csv');prior=registry.read_bytes()
with registry.open() as f:header=next(csv.reader(f))
res=s['pooled'];vals=['203 / CNN3→6 고정seed확대','202분산진단 및 부모명시승인','CNNseed동일평균확대가021오차와추정변동을줄이는가','052/088은DINOhead;018/021/133은CNN3seed.203은새3seed추가·기존후단고정','기존3seed021 vs 기존3+신규3동일mean6','새seed20262003/23/43+fold;45fits;Tiny12880epoch800updatesD4MSEAdamW.002→.0001wd.01last5;P021+.243*slope*(mean6-mean3)','pooledgain>0/세분할비악화/조건부CI하한>0/top5제거gain>0/분할예측분산비증가; 기술검증필수',status,f"pooled021 {res['reference_RMSE']:.10f} / mean6 {res['primary_RMSE']:.10f}; gain{res['gain']:.10f}; CI{res['conditional_CI95']}; variance {variance['partition_prediction_variance_reference']:.6f}->{variance['partition_prediction_variance_primary']:.6f}",'같은400ID3분할cluster10000draw;후단old3의존성유지;seed분산/n은가정기반proxy;historicalselection미보정','all6보존;seed선별/추가seed/비중탐색없음;full500/test/CSV/업로드현재범위없음','experiments/cnn-seed-expand-203/plan.json;study-complete.json;validation.json;summary.json;score-validation.json;interpretation.json;completion.json;reports/cnn-seed-expand-203-summary.json',now]
assert len(header)==len(vals);buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\n').writerow(vals)
with registry.open('ab') as f:
 if prior and not prior.endswith(b'\n'):f.write(b'\n')
 f.write(buf.getvalue().encode())
assert registry.read_bytes().startswith((_repo_path(O, 'strategy-experiment-register.before.csv')).read_bytes())
lines=['# 203 CNN3→6 seed 고정 대조 완료 — best021 보존','',f'신규3seed를 사전 고정해 원본+기존두재분할에서 **45회 학습**했습니다. 기존3seed를 포함한6개를 모두 동일평균했고 후단calibration/residual은 고정했습니다. 판정: **{status}**.','','| 분할 | best021 | CNN mean6 치환 | gain |','|---|---:|---:|---:|']
for split,row in s['by_partition'].items():lines.append(f"| {split} | {row['reference_RMSE']:.5f} | {row['primary_RMSE']:.5f} | {row['gain']:+.5f} |")
lines.extend([f"| pooled | {res['reference_RMSE']:.5f} | {res['primary_RMSE']:.5f} | {res['gain']:+.5f} |",'',f"400ID 묶음 조건부95%CI **[{res['conditional_CI95'][0]:.6f}, {res['conditional_CI95'][1]:.6f}]**. 같은400장의 반복분할이며 독립1200장이 아닙니다. 분할간 예측분산 **{variance['partition_prediction_variance_reference']:.6f}→{variance['partition_prediction_variance_primary']:.6f}**. 기존3seed에맞춘후단을유지한출력치환이며완전nested재학습결과가아닙니다.",'',f"기존45개+신규45개모델재로딩검증;신규MPSexact,CPU최대차{v['max_CPU_MPS_error']:.3g},독립앙상블수식차{v['max_independent_fullensemble_error']:.3g},10000bootstrap검증통과. 신규36,000updates;기존모델중복학습0. 모든seed보존,선별/구제비중탐색없음.",'','현재단계에서설치·다운로드·외부AI·holdout/test·전체500·새제출CSV·업로드없음. 비용0. 기존best021과모든보호파일유지. 결과:`reports/cnn-seed-expand-203-summary.json`, 계획·코드·증빙:`experiments/cnn-seed-expand-203/`.',''])
(_repo_path(R, 'README.md')).write_bytes(('\n'.join(lines)+'\n').encode()+(_repo_path(O, 'README.before.md')).read_bytes())
for p,h in {**plan['inputs_sha256'],**plan['protected_sha256']}.items():assert H(_repo_path(R, p))==h,p
W(_repo_path(O, 'preservation.json'),{'inputs_unchanged':len(plan['inputs_sha256']),'protected_unchanged':len(plan['protected_sha256']),'old45checkpoints_unchanged':True,'best021_sha256':H(_repo_path(R, 'artifacts/submission-candidate-021.csv')),'registry_original_bytes_preserved':True,'README_original_bytes_preserved':True,'registry_new_sha256':H(registry),'README_new_sha256':H(_repo_path(R, 'README.md'))})
W(_repo_path(O, 'completion.json'),{'study':203,'complete':True,'strong_development_gate':strong,'current_best':'021','best021_preserved':True,'new_fits':45,'no_full500_test_submission':True,'UTC':now,'evidence_sha256':{f:H(_repo_path(O, f)) for f in ['plan.json','code-lock.json','study-complete.json','baseline-validation.json','validation.json','summary.json','score-validation.json','interpretation.json','preservation.json']},'report_sha256':H(_repo_path(R, 'reports/cnn-seed-expand-203-summary.json'))})
manifest={str(p.relative_to(O)):H(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json'}
W(_repo_path(O, 'artifact-manifest.json'),manifest)
print(json.dumps({'study203_completed':True,'strong_gate':strong,'protected_inputs':len(plan['inputs_sha256']),'protected_files':len(plan['protected_sha256']),'artifact_files':len(manifest),'resources':interpretation['resources'],'best021_sha256':H(_repo_path(R, 'artifacts/submission-candidate-021.csv'))},ensure_ascii=False))
