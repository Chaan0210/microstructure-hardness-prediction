# 실험 자료

폴더를 연구 단계별로 묶었다. 실험 내부의 코드·설정·결과·로그는 같은 폴더에 두고, 최종 제출의 독립 실행 패키지는 [final/package](../final/package/README.md)에서 사용한다. [기존 연구 흐름과 결론](RESEARCH_HISTORY.md)도 보존했다.

| 분류                                                                         | 내용                                                   | 실험 폴더 수 |
| ---------------------------------------------------------------------------- | ------------------------------------------------------ | -----------: |
| [01-representations](./01-representations/README.md) · 표현·백본·학습 정책   | 초기 앙상블 재현, CNN/DINO 표현, 학습 정책과 검증 설계 |           45 |
| [02-microstructure](./02-microstructure/README.md) · 미세조직 측정·물리 특징 | Cellpose, U-Net, 결정립·방향·음영 특징과 회귀 파트너   |           10 |
| [03-robustness](./03-robustness/README.md) · 잡음·증강·일반화                | 열화 증강, 특징 개선, 정보량·분포·강건성 실험          |           15 |
| [04-residual-cnn](./04-residual-cnn/README.md) · 잔차 CNN·후속 검증          | 잔차 보정, 후보 076–083 계열과 재검증                  |            9 |
| [05-source-flow](./05-source-flow/README.md) · 최종 source-flow 모델         | 최종 제출 #62로 이어진 source encoder와 readout        |            1 |
| [frameworks](./frameworks/README.md) · 공용 실행 환경                        | TabPFN·TabICL 등 공용 도구·가중치·환경                 |            4 |
| [submissions](./submissions/README.md) · 제출 작업 기록                      | 실험 결과와 구분한 제출 과정의 원자료                  |            2 |

## 이름과 파일 배치

- 번호가 있는 실험: `<4자리 study 번호>-<소문자 주제>`. 예: `0177-dinol`, `0300-cellpose-finetuning`.
- 날짜로 관리한 후속 실험: `<YYYYMMDD>[-HHMM]-<주제>`. 후보 번호는 `candidate-076`처럼 명시한다.
- 기존 실험 ID는 과거 기록의 식별자다. 각 분류의 표에 현재 폴더와 함께 적었다.
- 새 실험은 해당 단계 폴더에 만들고, 설명은 `README.md`, 결과 요약은 `summary.json`, 검증은 `validation.json`, 실행 로그는 `run.log`를 기본 이름으로 쓴다.
- 기존 산출물의 파일명·과학적 내용은 보존했다. 실패한 대조군과 검증 자료도 재사용 가능한 실험 근거로 취급한다.

## 재현과 경로

`research_paths.py`는 과거 ID로 기록된 경로를 현재 위치로 연결하고 초기 study 모듈의 import 경로를 설정한다. 보존된 결과 JSON의 경로·체크섬은 원래 실행의 기록이며, 최종 패키지에는 이 호환 모듈이 필요하지 않다.

아래 분류의 **★**는 최종 패키지의 의존성 목록에 해당 실험의 파일이 포함돼 있음을 뜻한다. 전체 파이프라인을 다시 학습하려면 별도 데이터·환경·중간 자산이 필요한 과거 실험도 있다.
