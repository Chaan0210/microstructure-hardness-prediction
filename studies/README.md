# 초기 study 소스

초기 단일 스크립트 417개를 study 번호 구간과 역할로 나눴다. 파일명과 Python 모듈 이름은 유지해 기존 import와 저장 모델의 클래스 이름을 보존한다.

고전 영상 특징·KRR·TinyCNN에서 DINOv2와 중첩 잔차 앙상블로 진행한 9/30–10/06 연구이며, 후보 018/021(Public 약 11.49)까지의 소스다. 이후 연구는 `experiments/`에 있다.

| 폴더 | 내용 | Python 파일 수 |
|---|---|---:|
| [baseline](baseline/README.md) | 첫 기준선 실험·공용 특징·모델 모듈 | 42 |
| [candidates](candidates/README.md) | 후보 생성·최종화·독립 검증 | 46 |
| [026-049](026-049/README.md) | study 026–049 | 37 |
| [050-074](050-074/README.md) | study 050–074 | 64 |
| [075-099](075-099/README.md) | study 075–099 | 64 |
| [100-124](100-124/README.md) | study 100–124 | 64 |
| [125-149](125-149/README.md) | study 125–149 | 52 |
| [150-174](150-174/README.md) | study 150–174 | 48 |
| [src](src/) | 첫 기준선의 `microstructure`·`microstructure_v2` 패키지 | 7 |

## 실행

저장소의 `.venv/bin/python`으로 해당 파일을 실행한다. 각 스크립트의 경로 설정이 다른 번호 구간의 모듈도 찾는다. 새 연구 코드는 이 보관 영역에 추가하지 않고 [experiments](../experiments/README.md)에 둔다.

```sh
.venv/bin/python -B studies/baseline/frozen_validation.py --help
.venv/bin/python -B studies/baseline/validate_submission.py --help
```

공유 데이터·산출물은 기존 상대 링크로 접근한다: `data → ../data`, `artifacts → ../artifacts`, `reports → ../reports`, `.venv → ../.venv`. 파일 경로의 논리적 기준은 `studies/`로 유지한다.

일부 과거 후보 파이프라인은 보관되지 않은 `vendor/dinov2-official/provenance.json`이나 중간 특징 캐시를 import·실행 시 요구한다. 폴더 정리는 이 자료를 새로 만들거나 과거 실행의 체크섬을 덮어쓰지 않는다. 최종 제출 재현은 아래의 독립 패키지를 사용한다.

플랫폼 제출 번호, 후보 번호, study 번호는 서로 다르다. `050-074/dinob_062.py`는 DINO-B study 062이며 최종 제출 #62가 아니다.

결과는 [reports](../reports/README.md), 시간순 설명은 [연구 일지](../docs/research-log.md), 최종 재현은 [final/package](../final/package/README.md)에서 찾는다.
