# 최종 모델 실행 안내

대회 최종 수상 순위는 **3위**다. Public·Private 리더보드와 최종 수상 순위는 구분한다.

최종 제출 파일은 `candidate-source-flow38-boostedreadout.csv`이며, Public RMSE는 8.8196272229, Private RMSE는 9.6966884828이다. 이 문서는 학습·특징 생성·추론 코드와 실행 설정을 포함한다.

심사용 실행 경로는 제출 당시 가중치를 사용하는 `raw.py`다. 공식 원본 이미지에서 특징과 예측을 다시 계산한다. `train.py`는 대회 학습 데이터로 가중치를 새로 만드는 경로다.

## 파일 배치

코드와 자산의 `final/` 폴더를 하나로 합친다. `microstructure-solution.zip`과 `microstructure-assets.zip`으로 받았다면 같은 위치에 풀면 된다. 아래 환경 설치·실행 명령은 합쳐진 `final/`에서 실행한다.

GitHub에서 받은 코드의 실행 위치는 `final/package/`다. 공식 데이터와 공식 사전학습 가중치에서 시작할 때는 아래의 「학습 데이터부터 재학습」 절차를 사용한다. 학습 라벨·이미지 입력 캐시가 포함된 자산 ZIP은 공개 대상에서 제외하며, 제출 당시 가중치 추론 절차는 자산을 별도로 보유한 환경에서 사용한다.

공식 데이터는 다음 상대 경로에 둔다. 대회 데이터는 별도로 준비한다.

```text
data/train.csv
data/train/TRAIN_*.png
data/sample_submission.csv
data/test/TEST_*.png
```

## 실행 환경과 설치

개발·검증 환경은 macOS 27.0.1 arm64, Apple M1 Pro, Python 3.13.3이다. 전체 학습·원본 이미지 추론에는 Apple MPS가 필요하다. TabPFN의 기본 backend는 CPU이며, `--tab-device mps`는 TabPFN 부분의 실행 장치만 바꾼다.

세 환경의 라이브러리 버전은 각각 `requirements-core.txt`, `requirements-cellpose.txt`, `requirements-tabpfn.txt`에 고정되어 있다.

```sh
python3.13 -m venv .venv-core
python3.13 -m venv .venv-cellpose
python3.13 -m venv .venv-tabpfn
.venv-core/bin/python -m pip install -r requirements-core.txt
.venv-cellpose/bin/python -m pip install -r requirements-cellpose.txt
.venv-tabpfn/bin/python -m pip install -r requirements-tabpfn.txt
```

## 제출 당시 가중치로 재현

코드와 자산을 합치고 공식 데이터를 배치한 뒤 사전 검사를 실행한다.

```sh
.venv-core/bin/python -B raw.py --data data --check \
  --python-core .venv-core/bin/python \
  --python-cellpose .venv-cellpose/bin/python \
  --python-tabpfn .venv-tabpfn/bin/python
```

다음 명령은 예측을 `reproduced.csv`에 저장한다. 출력 파일이 이미 있으면 다른 이름을 지정한다.

```sh
.venv-core/bin/python -B raw.py --data data --out reproduced.csv \
  --python-core .venv-core/bin/python \
  --python-cellpose .venv-cellpose/bin/python \
  --python-tabpfn .venv-tabpfn/bin/python
```

전체 실행은 train/test 물리 특징, G018·DINO-L test 특징, source-flow train/test 특징, 117개 TabPFN context, 잔차 CNN 입력과 최종 결합을 계산한다. 저장된 참조 특징·예측과 비교하며, 중간 작업 폴더는 종료 시 삭제한다.

## 학습 데이터부터 재학습

코드와 공식 데이터로 시작한다. `prepare_pretrained.py`는 공식 사전학습 가중치를 다운로드하고 해시를 검사하므로 이 준비 단계에는 네트워크 연결이 필요하다.

```sh
.venv-core/bin/python -B prepare_pretrained.py --download
.venv-core/bin/python -B train.py init --workspace trained --data data \
  --python-core .venv-core/bin/python \
  --python-cellpose .venv-cellpose/bin/python \
  --python-tabpfn .venv-tabpfn/bin/python
.venv-core/bin/python -B train.py all --workspace trained
.venv-core/bin/python -B trained/raw.py --retrained --data data --out retrained.csv \
  --python-core .venv-core/bin/python \
  --python-cellpose .venv-cellpose/bin/python \
  --python-tabpfn .venv-tabpfn/bin/python
```

모델 구성과 학습 단계는 `REPRODUCTION.md`에 있다. 학습 자식 프로세스에는 test 이미지·test 특징·sample submission 읽기를 차단하는 guard가 적용된다.

## Markdown만 받은 경우

아래 `restore.py` 파일 블록의 Python 내용을 UTF-8 파일 `restore.py`로 저장하고, `microstructure-solution.md`와 같은 폴더에 둔다. 해당 폴더에서 다음 명령을 실행하면 코드·설정·실행 안내가 새 `final/`에 복원된다.

```sh
python3.13 restore.py microstructure-solution.md --out final
cd final
```

`--out`에는 아직 없는 디렉토리를 지정한다. 복원 후 위의 환경 설치와 실행 절차를 따른다. 제출 당시 가중치 재현에는 자산 폴더도 합쳐야 한다.

## 모델 출처와 AI 활용

사전학습 모델의 공식 출처·버전·해시는 `model-provenance.json`, 라이선스와 고지는 `licenses/` 및 각 모델의 `project/experiments/*/licenses/`에 있다.

코드 분석·작성·정리와 로컬 실행·검증에 AI 도구를 활용했다.
