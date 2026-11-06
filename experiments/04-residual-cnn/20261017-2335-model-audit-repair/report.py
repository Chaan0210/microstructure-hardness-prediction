"""Close a rejected fixed experiment only after all verification has finished."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,csv,hashlib,datetime
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
def read(name):return json.loads((_repo_path(HERE, name)).read_text())
def write(name,data):
 with (_repo_path(HERE, name)).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 r=read('retarget-results.json');l=read('log-results.json');d=read('dense-results.json');dd=read('dense-audit-decision.json');pres=read('preservation.json');rep=read('reproduction/082-inference-verification.json')
 assert not r['advance'] and not l['advance'] and not dd['advance'],'A candidate qualified: freeze and complete its full500 protocol before closure'
 assert pres['all_unchanged'] and rep['all_passed'] and rep['CSV_byte_identical']
 for fn in ['teacher-source-verification.json','training-repeat-verification.json','log-model-verification.json','assembly-verification.json','legacy-context-verification.json']:assert read(fn)['passed'],fn
 dv=read('dense-training-verification.json');assert dv['operator_checks_passed']
 rows=[dict(protocol='baseline',arm='082',overall_rmse=9.256738746625928,high_rmse=11.508128300723715,overall_gain=0.)]
 for tag,z in [('retarget',r),('log',l),('dense',d)]:
  for arm,m in z['results'].items():rows.append(dict(protocol=tag,arm=arm,**{k:v for k,v in m.items() if not isinstance(v,(list,dict))},ci95=str(m['ci95']),family_ci=str(m['family_ci']),fold_gains=str(m['fold_gains'])))
 cols=list(dict.fromkeys(k for row in rows for k in row))
 with (_repo_path(HERE, 'comparison.csv')).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 decision=dict(new_candidate_recommended=False,incumbent='082',rollback='076',official_best_changed=False,submitted=False,hypotheses={'teacher_alignment':r['advance'],'log_retransformation':l['advance'],'dense_sampling':dd['advance']},data_scope='Train-only fitting and selection; incumbent conservation replay only reads deployed test inputs',validation_scope='Reused400labels with nested new fitting, conditional on historical frozen representations/selection',completed=True)
 write('decision.json',decision)
 elapsed={f:read(f)['seconds'] for f in ['teachers-complete.json','cnn-retarget-complete.json','log-complete.json','dense-resource-probe.json','dense-mps-resource-probe.json','dense-complete.json','training-repeat-verification.json','dense-training-verification.json','log-model-verification.json','legacy-context-verification.json','audit.json','log-matched-size-diagnostic.json','calibration-delete-one.json']}
 # Include wall spans of the commands not covered by stage timers, if available.
 if (_repo_path(HERE, 'verification-stage-times.json')).exists():
  extra=read('verification-stage-times.json');elapsed['other_verification_command_overhead']=sum(z['seconds'] for z in extra if z['stage'] not in ['repeat','dense-repeat','log-models','legacy','log-size','calibration'])
 elapsed['failed_startup_misc_margin']=60
 compute=sum(elapsed.values());assert compute<5400,('new compute budget',compute)
 write('compute-accounting.json',dict(stage_seconds=elapsed,accounted_compute_seconds=compute,budget_seconds=5400,within_budget=True,definition='New fitting, timing probes, inference/diagnostics and verification. Reading, authoring and preservation hashing excluded. Misc60s margin covers short failed startup/evaluation commands. This is not a hardware speed benchmark.'))
 m=r['results'][r['primary']];h=l['results']['smear'];v=d['results']['pool_matched'];p=d['results']['pick_matched'];old=r['results']['old_matched'];matched=r['primary_vs_control'];size=read('log-matched-size-diagnostic.json');cal=read('calibration-delete-one.json');maxjump=max(z['largest_change'] for z in cal['rows'])
 table='\n'.join(f"| {name} | {x['overall_rmse']:.5f} | {x['high_rmse']:.5f} | {x['overall_gain']:+.5f} |" for name,x in [('기존 CNN, 076 내부 계수 재선택 대조',old),('076 교사로 재학습',m),('로그 역변환 보정',h),('촘촘한 합성곱+간격4 선택 대조',p),('촘촘한 합성곱+4×4 평균',v)])
 cases=read('influential-cases.json')['cases']['dense'][:3]
 case_table='\n'.join(f"| [{x['ID']}]({ROOT}/data/train/{x['ID']}.png) | {x['hardness']:.3f} | {x['matched_oldCNN']:.3f} | {x['candidate']:.3f} | {x['SSE_gain_vs_matched']:+.3f} |" for x in cases)
 text=f'''# 모델 전체 재점검과 세 가지 수정 검증

**이번 점검에서도 082를 교체할 만큼 검증된 개선은 확보하지 못했다.** 미완료 교사 정합 검증을 완주하고, 로그 역변환과 첫 합성곱의 다운샘플링까지 구현해 비교했다. 새로운 제출·최종 파일 변경은 없으며 082/076을 보존했다.

| 방법 | 전체 400 RMSE | 고잡음 124 RMSE | 082 대비 개선량 |
|---|---:|---:|---:|
| 보존된 082 | 9.25674 | 11.50813 | — |
{table}

양수 개선량은 RMSE 감소다. 이 표는 Public 점수가 아니라 같은 400개 재사용 개발 이미지의 조건부 중첩 평가다. 모든 500개 라벨은 과거 노출되었고, 기존 U-Net/Cellpose 표현과 과거 선택은 새로 중첩 재학습하지 않았다. 여러 분할이나 부트스트랩이 새 독립 확인을 만들어 주지 않는다.

## 무엇을 실제로 확인했나

학습 CSV와 이번에 사용한 캐시의 ID/라벨 정렬, 500개 원본의 중복 여부, 교사 30개 묶음의 학습/질의 분리와 라벨 해시, 실패행 대체 규칙, 정규화·추론·앙상블 조립을 추적했다. 이미지 해시는 500개 모두 달랐다. 새 코드의 외부 라벨 사용 오류나 기본 라벨 순서 오류는 발견하지 못했다. 코드와 근거 위치는 AUDIT.md에 있다.

**교사 불일치는 존재하지만, 수정만으로 성능이 좋아지지 않았다.** 082의 CNN은 G021을 제외한 073 분기를 0.84로 정규화한 교사의 잔차를 배운다. 기존 T076 실험에서 완료된 교사 27개와 부분 계산 36개를 확인하고, 부족한 TabPFN 계산 180개를 완성했다. 기존 8개 CNN을 재사용하고 12개를 추가 학습해 5fold/2seed 검증을 끝냈다. 여전히 G021을 제외한 교사이며, 142/213/320 등의 학습 크기 차이까지 제거한 실험은 아니다.

076 교사 재학습의 082 대비 평균 이득은{m['overall_gain']:.5f}지만 95% 구간은{m['ci95']}다. 같은 076 내부 계수 선택을 적용한 **기존 CNN보다 {-matched['overall_gain']:.5f} 나빴다**. 기존 계수를 고정하면 재학습 RMSE는{r['results']['retarget_frozen']['overall_rmse']:.5f}로 악화했다. 이전 2fold의 일치 대조 이득+.02511은 나머지 3fold에서−.04243으로 뒤집혔다. 전체 이득이 큰 10개를 빼면 082보다 {-m['top10_removed_gain']:.5f} 나빠진다. 큰 평균 이득을 새로운 영상 신호로 해석할 수 없다.

**계수 선택은 작은 내부 표본에 민감하다.** 각 B에는 고잡음 24~41개만 있다. fold4는 기준 예측의 고잡음 RMS 1.2575 변화와 함께 선택이 global/0에서 분리/1로 바뀌었다. 내부 라벨 한 개을 제외하는 진단의 최대 계수 변화는{maxjump:.5f}였다. 선택 손실은 저잡음 보정까지 평가하지만 실제 적용은 고잡음에만 하는 불일치도 있으며, 실제 적용 기준으로 계산하면 3/5fold의 선택 강도가 바뀐다. 이 수정 방향은 기존 calibration_order/역할 검증에서도 이미 불안정했으므로, 이번의 새 승자로 재포장하지 않았다. 큰 grain-set 오차도 실패행 대체 후 실제로 전달되는 크기와 구분했다.

**로그 역변환 보정은 주 병목을 해결하지 못했다.** 원본 소스별 중첩 OOF로 추정한 배율은.9872~.9961이었다. 이 데이터에서는 단순한 일괄 과소예측 가정이 맞지 않았다. 새 보정은 전체 {-h['overall_gain']:.5f},고잡음{-h['high_gain']:.5f} 악화했고, 같은 데이터의 상수 오프셋 대조보다도 나빴다. 학습 크기를 256개로 맞춘 저장 모델 진단도 개선량 {size['results']['smear256']['gain_over_same_size_raw']:+.5f}였다. 단순 퍼짐/평균 보정으로 새 신호를 얻지 못했다.

**첫 다운샘플링을 바꾼 학습도 검증했다.** 기존 7×7/stride4를 7×7/stride1로 촘촘히 계산한 뒤, 4간격 선택 또는 4×4 평균을 적용했다. 파라미터는 양쪽 68,788개이며 교사, 목표, 시드, 40epoch, 원본 D4는 동일하다. 변경한 것은 모델 내부의 집계 방식이다. 두 군 모두 같은 MPS 환경으로 총 40개 모델을 완료했다. CPU와 GPU의 학습 궤적 차이가 상쇄된다고 가정하지 않고, 같은 GPU의 대조군과 기존 CNN에 동일한 계수 선택을 적용한 대조군을 함께 봤다.

평균 풀링 군의 082 대비 이득은 {v['overall_gain']:+.5f}, 95% 구간은 {v['ci95']}, 세 가설을 고려한 98.33% 구간은 {v['family_ci']}다. 개선한 fold는 {v['improved_folds']}/5, 상위 10개 제외 이득은 {v['top10_removed_gain']:+.5f}다. 같은 촘촘한 계산의 선택 대조군 대비 이득은 {d['primary_vs_control']['overall_gain']:+.5f}, 동일 계수 절차의 기존 CNN 대비 이득은 {dd['versus_matched_old_CNN']['overall_gain']:+.5f}다. 사전 채택 기준과 추가 귀속 검사를 통과하지 못했다. 개별 seed, 고정 계수, 모든 대조군, 큰 손해 ID는 dense-results.json과 dense-per-image.csv에 보존했다.

원래 CNN의 1px 격자 진단 차이는 0.066~0.080HV RMS, 최대 0.212HV였다. 이것만으로 큰 격자 불안정성이 입증된 것은 아니다. 4×4 평균은 집계와 유효 수용 영역, 위치도 함께 바꾸므로 어떤 물리 구조가 달라졌는지 분리해 입증하지 못했다. 공유 오차나 이번 실패를 정보의 복구 불가능성으로 해석하지 않는다.

아래는 평균 풀링 변경으로 손해가 커진 사례다. 평가 후 선정한 설명용 사례이며 새 규칙이나 모델 선택에 사용하지 않았다. 원본 영상도 직접 확인했지만, 약하게 보이는 경계를 실제 결정립 정답으로 간주하지 않는다.

| 원본 | 실제 경도 | 동일 계수 절차의 기존 CNN | 평균 풀링 | 제곱오차 감소량 |
|---|---:|---:|---:|---:|
{case_table}

과소예측과 과대예측 사례가 함께 있어 일괄 평균 오프셋으로 모든 실패를 설명할 수 없었다.

## 검증·보존·실행

- 새 이미지 학습은 A213/B107 선택과 T320/V80 평가로 분리했다. 회귀 전처리, alpha, 보정은 해당 학습 partition만 사용했고 파생 사본은 원본과 그룹화했다. 모델, seed, epoch를 외부 점수로 고르지 않았다.
- 원래 CNN의 20개 문맥은 저장 예측을 정확히 재현했다. 교사 정합 CNN의 한 40epoch 전체 재학습은 학습 밖 라벨과 픽셀을 바꿔도 가중치와 원래 질의 예측 차이가 0이었다.
- 새 평균 풀링 CNN의 40epoch 재학습은 가중치 최대 차이 {dv['full_training_weight_maxdiff']:.9g}, 원래 질의 예측 최대 차이 {dv['full_training_original_query_maxdiff']:.9g}HV였다. 저장 모델 추론은 모든 40개 모델에서 검사했다. 재학습의 정확 일치 여부는 {dv['full_training_bit_exact']}이며 저장 가중치 재현과 구분한다.
- 로그 회귀 120개 저장 모델의 추론과 소스 제한 재적합, 독립 RMSE와 상위 기여 제외 계산을 확인했다. 조건부 부트스트랩은 모델 선택 역사나 공유 학습 집합의 의존성을 제거하지 않는다.
- 새 계산 합계는 약 {compute/60:.2f}분으로 90분 한도 이내다. CPU/MPS 프로브의 보수적인 3000초 게이트와 총예산의 불일치를 정확도 평가 전에 수정한 기록도 보존했다. 설치, 유료 자원 사용, 제출은 없다.
- **보호 파일 {pres['files']:,}개와 Git HEAD, 기존 tracked diff는 그대로다. 076 캐시 조립과 082 저장 가중치 추론 CSV가 원본과 byte 단위로 일치했다.** 이는 전체 과거 학습의 재실행을 뜻하지 않는다. 보존 검증에서만 기존 배포 test 입력을 읽었고 새 학습이나 선정에는 사용하지 않았다.

이번에 남긴 병목은 고잡음에서 안정적인 추가 설명력을 얻지 못하는 표현과, 작은 내부 표본의 보정 선택이다. 교사 학습 크기 변화의 모든 효과를 분리해 해결했다고 주장하지 않는다. 새 공식 최고 모델은 지정하지 않았고 082(Public 8.9343098075), 076(Public 8.9646534604)을 그대로 유지한다.

재현 명령은 [RUN.md]({HERE}/RUN.md), 사전 조건은 [PLAN.md]({HERE}/PLAN.md), 전체 수치는 [comparison.csv]({HERE}/comparison.csv)에 있다. 개별 결과와 실패 기록도 이 폴더에 남아 있다.
'''
 with (_repo_path(HERE, 'REPORT.md')).open('x') as f:f.write(text)
 write('completion.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision=decision,protected_files=pres['files'],incumbent076_082_byte_identical=True,compute_seconds=compute))
 print(json.dumps(dict(complete=True,new_candidate=False,compute_minutes=compute/60,protected_files=pres['files']),indent=2))
if __name__=='__main__':main()
