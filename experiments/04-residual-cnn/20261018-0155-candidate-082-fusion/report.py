"""Write the measured result, keeping a dev mean gain distinct from promotion."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,csv,datetime
import numpy as np
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((_repo_path(HERE, name)).read_text())
def save(name,value):
 with (_repo_path(HERE, name)).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
 r=read('results.json');s=read('screen-decision.json');v=read('verification.json');p=read('preservation.json');rep=read('reproduction/082-inference-verification.json');assert v['passed'] and p['all_unchanged'] and rep['all_passed'] and rep['CSV_byte_identical'];assert not s['advance'],'Run the frozen confirmation and production stages before closing a qualifying candidate'
 m=r['results']['primary'];shared=r['results']['shared'];quality=r['results']['quality_calibration'];fixed=r['results']['fixed'];z=np.load(_repo_path(HERE, 'oof.npz'));gain=(z['base082']-z['y'])**2-(z['primary']-z['y'])**2;fold_share=[float(gain[z['fold']==k].sum()/gain.sum()) for k in range(5)];dg=read('diagnostics.json');rowtext='\n'.join(f"| {name} | {r['results'][key]['overall_RMSE']:.5f} | {r['results'][key]['high_RMSE']:.5f} | {r['results'][key]['gain']:+.5f} |" for key,name in [('none','보존된082'),('fixed','고정 비율 결합'),('shared','기존CNN 보정만 재학습'),('quality_calibration','기존CNN 보정+품질 조건'),('primary','내부에서 선택한 결합 절차')]);casetext='\n'.join(f"| {x['ID']} | {x['hardness']:.3f} | {x['incumbent082']:.3f} | {x['fusion']:.3f} | {x['SSE_gain']:+.2f} |" for x in dg['cases']['harms'][:3])
 text=f'''# 082 기반 결합: 개발 평균 개선, 채택 조건 미달

**결합 프로토타입을 구현해 중첩 검증 평균은 낮췄다. 하지만 일반화 개선으로 확정하거나082를 교체할 근거는 확보하지 못했다.** 새CNN 세 종류와 경계·연결 모델을 함께 사용했고, 가중치·방법 선택은 내부B에서만 했다. 바깥 검증에서 잘 맞은 이미지나 분할을 골라 붙이지 않았다.

| 방법 | 전체400 RMSE | 고잡음124 RMSE | 082 대비 개선량 |
|---|---:|---:|---:|
{rowtext}

단일 주 비교 대상은 사전에 고정한 **내부 선택 절차**다. 바깥 점수가 가장 낮은 별도 실험군을 사후 선택하지 않았다. 주 비교의 전체 이득은 **{m['gain']:.5f}**, 고잡음 이득은 **{m['high_gain']:.5f}**이며 {m['improved_folds']}/5fold가 개선됐다. 전체 이득의 조건부95% 구간은 **[{m['CI95'][0]:+.5f}, {m['CI95'][1]:+.5f}]**다.

가장 큰 이득을 준10개 원본을 제외하면 **{-m['removed_top']['10']:.5f} 악화**한다. 마지막 fold가 순 제곱오차 감소의 **{100*fold_share[4]:.1f}%**를 차지했다. 따라서 평균 감소를 독립적인 일반화 성과로 주장하지 않는다. 이400개 라벨은 이미 여러 차례 사용됐고, 기존 표현·모델 가족 선택도 역사적 탐색의 영향을 받는다.

## 무엇을 합쳤나

기존082의CNN,076 교사로 다시 학습한CNN, 촘촘한 합성곱의 간격 선택CNN, 평균 풀링CNN, 경계·연결 학습의 물리Tab 보정 방향을 사용했다. 모델 가족마다 두seed를 동일 평균했다. 이전에 실패한 복제 입력 평균·단순 시드 불일치 감쇠와 달리, 이번에는 새로 완료된 서로 다른 분기들을 동시에 적합했다.

1. 고정 결합은082를 절반 유지하고, 나머지 네 분기를 각각1/8씩 더한다. 이 평균 이득은{fixed['gain']:.5f}뿐이었다. 평균의 제곱오차 항등식을 독립 확인했고, 서로 다른 예측의 분산이 줄여 준 양은{r['fixed_diversity']['disagreement_MSE']:.6f}HV²였다. 이 조합의 다양성 이득은 작았다.
2. 공동 결합은076 위 잔차 보정들을 동시에 적합하되, 기존082의CNN 계수를 최소 절반 유지하고 CNN 계수 합을1 이하로 제한했다. 경계 보정은0~0.5다. 복제 열을 넣어도 정규화가 느슨해지지 않도록 실제 예측 변화량에 벌점을 줬다.
3. 조건부 결합은 잡음 추정치와 log(1+분할 객체 수)에 따라 계수를 선형 변화시킨다. 정규화는 해당 내부 학습 원본만 사용하고, 품질 값은[-2,2]로 제한했다. 네 모서리에서 계수 제약을 걸어 질의 위치에서도 제약을 만족시켰다. 이 객체 수는 실제 결정립 정답으로 검증된 값이 아니다.

선택된 구성은fold0/1에서공동 결합λ1,fold2에서고정 결합,fold3에서조건부λ10,fold4에서조건부λ1이었다. 고잡음 외의 원본과 실패행은082와 정확히 같다. 모든 계수와 각 원본에서의 적용값은fold*-selection.json과coefficients.csv에 있다. 계수는 전체 예측의 지분이 아니라076 위에 더하는 잔차 보정의 계수다.

## 이득의 대부분은 어디서 왔나

기존CNN의 보정 크기만 바꾸는 대조군도 **{shared['gain']:.5f}**의 평균 이득을 냈다. 새 분기를 포함한 주 절차의 추가 이득은 **{r['primary_vs_controls']['shared']['gain']:.5f}**,95% 구간 **[{r['primary_vs_controls']['shared']['CI95'][0]:+.5f}, {r['primary_vs_controls']['shared']['CI95'][1]:+.5f}]**였다.

품질 조건을 기존CNN에만 제공한 더 엄격한 대조군 대비 추가 이득도 **{r['primary_vs_controls']['quality_calibration']['gain']:.5f}**,95% 구간 **[{r['primary_vs_controls']['quality_calibration']['CI95'][0]:+.5f}, {r['primary_vs_controls']['quality_calibration']['CI95'][1]:+.5f}]**로 불확실했다. 따라서 새 영상 정보가 안정적인 추가 설명력을 제공했다고 결론내리지 않았다. 비슷한 오차를 내는 분기들이 일부 큰 오차 사례를 보정하면서 평균은 좋아졌지만, 손해 사례도 남았다. 이것은 영상 정보가 본질적으로 복구 불가능하다는 증거가 아니다.

| 악화 사례 | 실제 경도 | 082 | 결합 | 제곱오차 감소량 |
|---|---:|---:|---:|---:|
{casetext}

반대로 TRAIN_000467과TRAIN_000209에서는 큰 이득이 있었다. 이런 사례를 평가 뒤 별도 예외 처리나 라우팅 규칙으로 사용하지 않았다. 전체 예측과 사례는per-image.csv,diagnostics.json에 저장했다.

## 검증의 범위와 보존

각 외부 fold는T320/V80이며, 결합 학습에는A213으로만 학습한 전문가의B107 예측을 사용했다. 다른 fold의OOF를 다시 나눠 메타 학습에 넣지 않았다. B의 한 원본을 LOO 평가할 때는082의 기존 계수, 새 계수, 품질 정규화까지 모두 그 원본을 제외해 다시 적합했다. 이후 모든B로 다시 적합하고V에서 한 번 평가했다. 공유된 과거 표현에 조건부인 평가이며, 새 독립 holdout은 아니다.

- 원본082 재구성은5개fold 모두 차이0이었다. 독립 active-set QP와15개공동 결합 해를 비교했고, 질의 예측 최대 차이는{max(x['query_prediction_maxdiff'] for x in v['QP_global_independent_active_set']):.8f}HV였다.
- 같은 예측을 복제한 수치 대조의 최대 차이는1.8e-9HV였다.30개 저장 결합 예측은 정확히 재현했다. 계수 제약, 원본별 RMSE, 상위 기여 제외 계산도 독립 검산했다.
- V라벨과 질의 특징을 다른 값으로 바꾼 파일을 실제 로더에 넣어도 전체 내부 선택이 정확히 같았다. 별도로B의 보류 라벨이 기존 계수·정규화·모든 후보 적합에 들어가지 않는 것을 검증했다.
- 보호 파일 **{p['files']:,}개**,GitHEAD와 기존tracked diff는 모두 그대로다.076 캐시 조립과082 저장 가중치 추론의CSV도 원본과byte 단위로 일치했다. 이는 전체 과거 학습 재현과 구분한다.
- 새 이미지 모델 학습, 설치, 유료 자원 사용, 제출은 없다. 보존 검증에서만 기존 배포test 입력을 읽었고, 새 학습이나 선택에는 사용하지 않았다.

사전 조건 중 평균 이득·고잡음 이득·개선fold 수는 통과했지만, 신뢰 구간과 상위10개 제거 조건은 실패했다. 따라서 예정된 동일 학습 크기 추가 검증과full500 제출 후보 제작 단계로 확대하지 않았다. **082/076은 유지하며, 이 프로토타입을 새 공식 최고나 제출 권고 후보로 지정하지 않는다.**

재현 명령은[RUN.md]({HERE}/RUN.md),사전 규칙은[PLAN.md]({HERE}/PLAN.md),전체 수치는[comparison.csv]({HERE}/comparison.csv)에 있다.
'''
 with (_repo_path(HERE, 'REPORT.md')).open('x') as f:f.write(text)
 save('decision.json',dict(primary_mean_dev_gain=m['gain'],primary_high_dev_gain=m['high_gain'],mean_dev_improved=True,validated_generalization_improvement=False,new_candidate_recommended=False,incumbent='082',rollback='076',official_best_changed=False,submitted=False,screen=s,fold_SSE_gain_share=fold_share))
 save('completion.json',dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protected_files=p['files'],new_image_fits=0,models='Five fold-specific nested fusion selectors plus controls; not a full500 deployment model',primary_overall=m['overall_RMSE'],primary_high=m['high_RMSE'],incumbent_saved_prediction_replay_byte_exact=True))
 print('FUSION COMPLETE: mean gain',m['gain'],'promotion',False)
if __name__=='__main__':main()
