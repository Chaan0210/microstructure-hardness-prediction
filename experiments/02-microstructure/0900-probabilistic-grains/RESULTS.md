# Study900: original-train probabilistic measurements and grain-set learning

**판정: 큰 폭의 개선은 확인되지 않았다. 가장 나은 고정 후보는 Ridge 20% 결합으로 dev 9.940240 → 9.831835다. 8점대 달성이나 Public/private 개선은 아직 확인하지 않았다.**

기존 코드와 제출물을 보존하고 별도 경로에서 실행했다. 추가 합성 데이터와 test 학습은 없다. 기존 400개 개발 ID의 3개 분할을 재사용한 조건부 검증이며 Public/private 점수는 측정하지 않았다.

| 방법 | dev RMSE | 045 대비 개선 | 고잡음·조대/중간 RMSE |
|---|---:|---:|---:|
| 045 reference | 9.940240 | +0.000000 | 13.524897 |
| 021 weight-only control | 9.903711 | +0.036529 | 13.467104 |
| Ridge 20% | 9.831835 | +0.108405 | 13.372118 |
| Ridge fixed021 share | 9.877356 | +0.062884 | 13.446991 |
| FGLS 20% | 9.834183 | +0.106057 | 13.380214 |
| Raw grain summary Ridge all3seed mean 20% | 10.069542 | -0.129302 | 13.621791 |
| Grain set all3seed mean 20% | 9.875014 | +0.065226 | 13.470890 |
| Grain set all3seed fixed021 share | 9.912919 | +0.027322 | 13.520850 |
| Bare X98 with matched stage2 selection 20% | 9.856435 | +0.083805 | 13.394312 |
| threshold__physical 20% | 10.550044 | -0.609804 | 14.149790 |
| threshold__x98_augmented 20% | 9.896082 | +0.044158 | 13.444328 |
| image_gmm__physical 20% | 10.293926 | -0.353686 | 13.919473 |
| image_gmm__x98_augmented 20% | 9.838171 | +0.102069 | 13.377660 |
| hierarchical__physical 20% | 10.442593 | -0.502353 | 13.882952 |
| hierarchical__x98_augmented 20% | 9.928435 | +0.011806 | 13.497491 |
| point_rbf 20% | 10.150810 | -0.210569 | 13.718027 |
| uncertain_rbf 20% | 10.152335 | -0.212095 | 13.683278 |
| PFN base98 20% | 9.866652 | +0.073589 | 13.383929 |
| PFN phase_mean 20% | 9.889573 | +0.050668 | 13.451333 |
| PFN phase_mean_and_sd 20% | 9.879261 | +0.060979 | 13.432074 |

## 결과 해석

- Ridge 보완의 일부 이득은 기존021 비중 감소에서 온다. 021 비중을 유지하면 개선은 약0.063이며, 단순 비중 조정 대조군에 대한 우위는 불확실하다.
- 집합 모델 최초seed 결합9.830188만 선택하지 않았다. 추가2seed를 모두 평균하면9.875014로 효과가 줄고, 고잡음 구간의 개선은 뚜렷하지 않았다. 따라서 전체500 재학습/추론으로 승격하지 않았다.
- 동일한 stage2 내부 CV를 적용한 bare-X98 대조군은9.856435, GMM 추가는9.838171, 계층 확률 추가는9.928435다. 회귀기 선택 절차 차이를 분리해도 계층 모형은 도움이 되지 않았다.
- MPS TabPFN은 기존98열만 쓴 비교가9.866652, 계층 분율 추가9.889573, 분율+불확실성9.879261이다. 강한 판독기에서도 새 확률 특징의 이득이 나타나지 않았다.
- 확률 입력 커널은 점 입력 커널과 같은 특징/내부 CV로 비교했고 개선되지 않았다. 이는 이 측정 모형과 커널 구현의 결과이며, 모든 확률적 방법이나 원본 영상의 정보 한계를 증명하지 않는다.
- 제출용 CSV로 생성하고 재현 검증한 후보는 Ridge 20%다. 기존045를 자동 교체하거나 온라인 제출하지 않았다.

## 검증과 해석

- 원본 500장과 고정 제로샷 마스크에서 결정립 관측을 추출했다. 개별 결정립은 독립 경도 표본으로 복제하지 않았다.
- 신규 측정 사전분포·스케일러·회귀기·신경망 조기종료는 outer training 안에서만 학습/선택했다.
- 기존 X98 및 045 자체는 과거에 개발한 표현이므로 완전히 새로운 검증이라고 부를 수 없다.
- 최초 확률 모델의 단일 최적점 적분은 다중 해를 놓쳐 무효 처리했다. 해당 파일은 삭제하지 않고 INVALID 표기로 보존했다.
- 최종 다중 최적점 적분은 독립 적분 기준과 비교했다. 공유 음영·분포 모형·공분산 단순화는 여전히 가정이다.
- 집합 모델은 최초 seed만 선택하지 않고 사전 고정한 추가 2개를 포함해 모두 평균했다.
- 작은 집합망은 벤치마크에서 CPU가 더 빨랐다. 대형 TabPFN 판독기는 MPS에서 실행했다.
- 최종 확률 특징:60캐시/20,400추론 전부 수렴, 학습 범위 검증. 드문6사례만 경계에 안전한 Beta-Jacobi 및 적응 적분을 사용했다.
- MPS의 phase+SD 저장 context는 같은 MPS에서80개 예측이 정확히 재현됐다. CPU 내 반복도 정확하나 CPU-MPS 사이 RMS0.028465HV 차이는 남는다(20%결합 RMS0.005693HV).
- 기존 코드·문서·제출물 21,263개 SHA256 재검사: 변경/삭제 0개.

## 실행 파일

- `ridge/run.py`: Ridge 및 FGLS 대조.
- `extract_objects.py`: 원본 화소 결정립 데이터.
- `probabilistic/contrast-adaptive/run_stage2.py`: 수정된 확률 측정.
- `set_model/run.py`, `set_model/run_replication.py`: 집합 모델 및 3seed 확인.
- `probabilistic/contrast-adaptive/run_stage4.py`: 점 입력/확률 입력 커널 대조.
- `pfn_readout.py`: MPS TabPFN 동일 조건 비교.
- `verify_mps_context.py`: 같은 MPS에서 저장 context 재현.
- `matched_ridge_control.py`: 동일 내부 CV의 bare-X98 귀속 대조.
- `ridge/finalize500.py`: 검증한 고정 Ridge 결합의 전체500 학습 및 로컬 추론.

전체 수치, 구간별 오차, 분할별 결과와 조건부 ID bootstrap은 `results.json`에 저장했다.
로컬 CSV는 `ridge/final500/candidate-045-ridge20.csv`에 있으며 제출하지 않았다.

MPS 비교를 새 출력 경로에 재현하는 명령(저장소 루트에서 실행):

```sh
experiments/frameworks/tabpfn-v2/.venv/bin/python -B experiments/02-microstructure/0900-probabilistic-grains/pfn_readout.py --features experiments/02-microstructure/0900-probabilistic-grains/probabilistic/contrast-adaptive/stage2/features --output experiments/02-microstructure/0900-probabilistic-grains/pfn-readout-reproduction --device mps
```

현재 진입점은 무효 처리한 원래 캐시와 완료된 결과 덮어쓰기를 거부한다. 실제 실험 당시 소스는 `pfn_readout_executed.py`에 SHA 그대로 보존했고, 기본 경로/입출력 보호 변경은 `pfn-readout-v2/entrypoint-repair.json`에 기록했다. 모델 생성과 특징 조립의 AST는 동일하다.
