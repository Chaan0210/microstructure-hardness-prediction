# studies/src/

대회 첫날(9/30)에 만든 첫 기준선 파이프라인이다. 이미지마다 형태 특징을 뽑아 Ridge·ExtraTrees·HGB 등으로 회귀한다. 최종 모델에는 쓰이지 않았다.

| | `microstructure` (v1) | `microstructure_v2` |
|---|---|---|
| 특징 | `features.py`: `morphology-v1` | v1 특징 + `extra.py`의 추가 특징 = `morphology-v2`. 밝기 임계 마스크(Otsu, 사분위)의 영역 통계, granulometry(white/black-hat 5개 크기), 경계 밀도, 국소 대비 |
| 모델 후보 | 평균 기준선, Ridge(α=100), ExtraTrees(leaf 2·4), HGB | v1 후보 + Ridge(α=1000) |
| 홀드아웃 | 개발 CV로 고른 뒤 홀드아웃 RMSE와 이미지별 오차 CSV를 낸다 | 홀드아웃 평가를 미룬다. v1 실행(run-001)에서 홀드아웃을 이미 봤기 때문에 이후 선택은 개발 CV로만 한다 |
| 사용처 | `studies/baseline/run.py`, `improve.py`, `research_grains.py`, `tests/test_pipeline.py` | `studies/baseline/run_v2.py` 등 초기 스크립트 |

원래 저장소 루트의 `src/`에 있었다. 두 패키지 모두 `pipeline.py`의 `ROOT`로 저장소 루트를 잡고 프로젝트 밖 경로는 거부한다. 이 폴더로 옮기면서 `parents[2]`를 `parents[3]`으로 바꿔 같은 루트를 가리키게 했다.
