# Finalize202 evidence; no model fitting or analysis change.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv,datetime,hashlib,io,json
from pathlib import Path
R=_RepoPath(__file__).resolve().parents[3];O=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def J(p):return json.loads(p.read_text())
def W(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
plan=J(_repo_path(O, 'plan.json'));s=J(_repo_path(O, 'summary.json'));v=J(_repo_path(O, 'validation.json'));seed=J(_repo_path(O, 'seed-variance.json'))
assert v['passed'] and v['summary_sha256']==H(_repo_path(O, 'summary.json'))
assert not s['passes_numeric_gate']
assert H(_repo_path(R, 'reports/strategy-experiment-register.csv'))==H(_repo_path(O, 'strategy-experiment-register.before.csv'))
assert H(_repo_path(R, 'README.md'))==H(_repo_path(O, 'README.before.md'))
for p,h in {**plan['input_sha256'],**plan['protected_sha256']}.items():assert H(_repo_path(R, p))==h,p
now=datetime.datetime.now(datetime.timezone.utc).isoformat();pct=100*(1-s['mean_across_partition_prediction_variance']['B_fixed']/s['mean_across_partition_prediction_variance']['R'])
interpretation={
 'study':202,'UTC':now,'status':'Completed fixed OOF experiment; primary simplification rejected; best021 retained',
 'actual_work':'Two predeclared current021 simplifications evaluated across all3existing5fold partitions with independent10000cluster-bootstrap verification. New training0; executed prediction-only comparison, not review alone.',
 'no_new_search':'A diagnostic intermediate cannot be promoted. B sole primary failed. No partial residual weight, alternate calibration, weights,seed subset or architecture search after outcomes.',
 'primary_result':{'pooled_RMSE_R':s['pooled_RMSE']['R'],'pooled_RMSE_B':s['pooled_RMSE']['B_fixed'],'gain':s['contrasts']['B_fixed']['pooled_RMSE_gain'],'conditional_CI95':s['contrasts']['B_fixed']['conditional_IDcluster_CI95'],'all3_partitions_worsened':True,'positive_folds_by_partition':[s['contrasts']['B_fixed']['per_partition'][p]['positive_folds'] for p in plan['data']['splits']],'prediction_partition_variance_R':s['mean_across_partition_prediction_variance']['R'],'prediction_partition_variance_B':s['mean_across_partition_prediction_variance']['B_fixed'],'variance_reduction_percent':pct},
 'reasoning_ko':[
  '현재021의 L/short-B를 그대로 두고 residual만 제거한 A, 이어서calibration도제거한 B를 실제 계산했다. 둘다 세분할 평균RMSE가 악화했다. 설명용 A를 대체후보로 고르지 않는다.',
  'B의 분할간 예측분산은 약25.6% 줄었다. 그러나RMSE는.153225 악화했다. 후단은 변동뿐 아니라 유효한 예측 보정도 제공한다. 이번 단순화로 정확도가 좋아졌다고 주장하지 않는다.',
  'B가 평균제곱오차 기준 개선한 이미지는218/400이지만 전체순이득은음수다. 개선 이미지수나특정fold만보고채택하지 않는다. pooled상위5수혜제거후gain은-.208373이다.',
  '조건부gain95CI는[-.324358,.006755]로0을포함한다. 재사용400장과고정모델조건의 불확실성이다. 새로운모집단에서B가유의하게나쁘다고단정하지않지만 사전운영게이트는실패했다.',
  'Seed분해는 현재021의 실제fold별slope와최종가중치를반영했다. rawseed분산의반복계산을넘어최종직접경로로전달되는양을측정했으나후단재학습변동은포함하지않는다.',
  '독립seed성분 가정에서 mean3의 조건부MSE분산 proxy는CNN .476614 / S .031447 / B .025805 / L .007880이다. 이부분합의약88%는CNN이다. L부터seed를늘리자는외부리뷰제안은현재가중치하의진단근거가약하다.',
  'L3→9의조건부MSE감소proxy는.005254이며 실제RMSE개선을예측하거나보증하지않는다. 모든head합proxy.361164도성분독립가정과3seed추정에의존한다. 이를기준으로새학습을자동개시하지않았다.',
  '3→9의기대분산법칙은이미선택된실현3seed묶음보다좋아진다는보증이아니다. CNN1→3전체nested대조는133에서이미수행되었다. 이번결과는새CNNseed학습이나닫힌CE분기를자동정당화하지않는다.'
 ],
 'seed_scope':{'conditional_mean3_MSE_proxy':{k:x['pooled_conditional_mean3_MSE_variance_proxy'] for k,x in seed['R']['components'].items()},'independent_sum':seed['R']['independent_component_sum_mean3_MSE_variance_proxy'],'unknown_covariance_bounds':seed['R']['covariance_agnostic_mean3_variance_bounds'],'CNN_fraction_of_independent_sum':seed['R']['components']['CNN']['pooled_conditional_mean3_MSE_variance_proxy']/seed['R']['independent_component_sum_mean3_MSE_variance_proxy'],'not_total_uncertainty':True,'no_new_seed_training':True},
 'duplicate_difference':'052/088/133 seed averaging and121/122 older ablations reused. New:021 residual/calibration-removal with L retained, current coefficient-propagated seedvariance, and across-partition sensitivity on same400 IDs.',
 'independent_validation':{'predictions_max_abs_error':v['independent_scalar_prediction_max_error'],'bootstrap_max_abs_error':v['independent_count_bootstrap_max_error'],'seed_identity_max_abs_error':v['independent_seed_variance_identity_max_error'],'three_repeated_measurements_per400_ID_cluster':True,'all_input_hashes_unchanged':True},
 'resources':{'scoring_seconds':s['seconds'],'validation_seconds':v['seconds'],'timing_scope':'Execution intervals only; excludes planning,coding,tool setup,reporting.','maximum_observed_RSS_bytes':max(s['RSS_bytes'],v['RSS_bytes']),'CPU_threads':2,'device':'CPU NumPy; no MPS training','new_fits':0,'cost':0},
 'scope':{'new_install':False,'download':False,'network':False,'holdout_test_reads':False,'new_labels':False,'full500_refit':False,'new_CSV':False,'upload':False,'best021_unchanged':True},
 'failures_preserved':'Initial read-allowlist blocked installed NumPy lazy import before new scores; fixed to permit read-only existing .venv, oldcode/failure/amendment preserved. Finalizer wrapper quote syntax failed before file creation; recorded. Plan/formulas/gates unchanged.',
 'decision':'Close current fixed simplification test; retain calibrated residual-corrected best021. No automatic alternative model/seed/weight experiment.',
 'remaining_blockers':'No execution blocker. Unknown generator and historical adaptive selection remain scientific limits.'
}
W(_repo_path(O, 'interpretation.json'),interpretation)
report={**s,'status':interpretation['status'],'validation_passed':True,'advance':False,'current_best':'021','interpretation':interpretation['reasoning_ko'],'seed_scope':interpretation['seed_scope'],'resources':interpretation['resources'],'evidence':'experiments/simplification-202/','plan_sha256':H(_repo_path(O, 'plan.json')),'validation_sha256':H(_repo_path(O, 'validation.json'))}
W(_repo_path(R, 'reports/simplification-202-summary.json'),report)
registry=_repo_path(R, 'reports/strategy-experiment-register.csv')
with registry.open() as f:header=next(csv.reader(f))
vals=['202 / 021 고정단순화·seed분산','AI 구조리뷰 적용 및 부모명시개선지시','후단residual와calibration 제거가현재021의추정불안정성을줄이면서오차도개선하는가','121/133 이전pipeline대조 재학습아님.021의L10%유지한고정제거와최종계수별seed분산전달 신규계산.052/088/133예측재사용','R=021; A=residual제거 설명용; B=residual+calibration제거 유일primary','같은400의기존3분할; B=.243KRR+.243CNNmean3+.162S+.162B+.09shortB+.1L; 가중치/seed탐색없음','B만 pooledgain>=.02/3분할양수/조건부CI하한양수/top5제외양수/분할간예측분산비증가; Afallback없음','완료·단순화기각·best021유지','pooled R12.81462814/A12.88177897/B12.96785309; Bgain-.15322494 CI[-.32435760,.00675490]; 분할예측분산3.37036→2.50828(-25.6%)','400ID묶음10000draw; 동일표본3회는독립1200아님.고정fit조건부CI·적응선택미보정.가중seedvariance는후단재학습/상호공분산제외','현단순화분기종료; 새학습/부분shrinkage/seed확대/holdout/test/CSV/업로드없음','experiments/simplification-202/plan.json;summary.json;seed-variance.json;validation.json;interpretation.json;completion.json;reports/simplification-202-summary.json',now]
assert len(header)==len(vals)
buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\n').writerow(vals)
prior=registry.read_bytes()
with registry.open('ab') as f:
 if prior and not prior.endswith(b'\n'):f.write(b'\n')
 f.write(buf.getvalue().encode())
assert registry.read_bytes().startswith((_repo_path(O, 'strategy-experiment-register.before.csv')).read_bytes())
intro='''# 202 AI 리뷰 적용: 고정 단순화 대조 완료 — best021 유지

저장된 개발400 OOF에서 현재021의 L/short-B 비중을 고정하고 residual 제거(A, 설명용), 이어서 calibration 제거(B, 유일 후보)를 사전 등록해 실제 비교했습니다. 새 모델 학습은 없습니다.

| 분할 | best021 | A: residual 제거 | B: residual+calibration 제거 |
|---|---:|---:|---:|
| 원본 | 12.78620 | 12.81763 | 12.91326 |
| 20261031 | 12.62190 | 12.71076 | 12.80088 |
| 20261101 | 13.03245 | 13.11357 | 13.18639 |
| pooled RMSE | 12.81463 | 12.88178 | 12.96785 |

B는 분할 간 예측분산을 **25.6%** 줄였지만 세 분할 모두 오차가 악화했습니다. gain **−0.153225**, 동일400ID 묶음 조건부95%CI **[−0.324358, 0.006755]**이며 사전 채택기준에 미달합니다. A도 채택하지 않습니다. 세 분할은 같은400장의 반복 측정이며 새 독립 검증이 아닙니다.

현재021의 직접 seed 경로만 분해한 조건부 mean3 MSE분산 proxy는 CNN **0.476614**, S **0.031447**, B **0.025805**, L **0.007880**입니다. 성분 독립·후단 고정 가정이며 전체 일반화불확실성이나 seed 확대의 실제 이득은 아닙니다. 052/088/133을 재학습하지 않았고 seed 확대를 자동 실행하지 않았습니다.

독립 스칼라 예측 재구성 최대차 **8.53e−14**, 전체10000 bootstrap 검증 최대차 **8.89e−15**. 입력209개·기존제출물 등50개 해시 동일. 계산1.51초/검증0.37초, 최대관측RSS 약196MB, CPU2, 비용0. 설치·다운로드·holdout/test·전체500·새CSV·업로드 없음. 초기 NumPy 지연 import 차단 오류와 수정도 보존했습니다.

계획·코드·결과·판정: `experiments/simplification-202/`, 요약: `reports/simplification-202-summary.json`. 현재 단순화 분기는 종료하고 best021을 보존합니다.

'''
(_repo_path(R, 'README.md')).write_bytes(intro.encode()+(_repo_path(O, 'README.before.md')).read_bytes())
for p,h in {**plan['input_sha256'],**plan['protected_sha256']}.items():assert H(_repo_path(R, p))==h,p
W(_repo_path(O, 'preservation.json'),{'inputs_unchanged':len(plan['input_sha256']),'protected_unchanged':len(plan['protected_sha256']),'best021_sha256':H(_repo_path(R, 'artifacts/submission-candidate-021.csv')),'registry_prior_bytes_preserved_as_prefix':True,'README_prior_bytes_preserved_after_new_intro':True,'registry_before_sha256':H(_repo_path(O, 'strategy-experiment-register.before.csv')),'registry_after_sha256':H(registry),'README_before_sha256':H(_repo_path(O, 'README.before.md')),'README_after_sha256':H(_repo_path(R, 'README.md'))})
W(_repo_path(O, 'completion.json'),{'study':202,'complete':True,'advance':False,'best021_preserved':True,'UTC':now,'new_training':0,'no_holdout_test_submission':True,'sources':{n:H(_repo_path(O, n)) for n in ['plan.json','run.py','validate.py','finalize.py','summary.json','validation.json','interpretation.json','preservation.json']},'report_sha256':H(_repo_path(R, 'reports/simplification-202-summary.json'))})
manifest={str(p.relative_to(O)):H(p) for p in sorted(O.iterdir()) if p.is_file() and p.name!='artifact-manifest.json'}
W(_repo_path(O, 'artifact-manifest.json'),manifest)
print(json.dumps({'completed':202,'advance':False,'best021_preserved':True,'files_hashed':len(manifest),'input_files_preserved':209,'protected_files_preserved':50,'registry_appended':True,'README_updated_with_backup':True,'best021_sha256':H(_repo_path(R, 'artifacts/submission-candidate-021.csv'))},ensure_ascii=False))
