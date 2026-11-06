from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,re
P=_RepoPath(__file__).resolve().parents[3];R=_repo_path(P, 'experiments/blur-augmentation-215');H=_repo_path(P, 'docs/records/roadmap-audit/handoff-after-215');H.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
names={213:'brightness-augmentation-213',214:'gamma-augmentation-214',215:'blur-augmentation-215'};summaries={n:read(_repo_path(P, 'experiments', name, 'summary.json')) for n,name in names.items()};audits={n:read(_repo_path(P, 'experiments', name, 'final-audit.json')) for n,name in names.items()};assert all(a['passed'] for a in audits.values());assert all(len(list((_repo_path(P, 'experiments', name)).glob('*/fit.json')))==7 for name in names.values());rows=[]
for n,label in [(213,'밝기'),(214,'감마'),(215,'블러')]:
 for key,scope in [('standalone','CNN 단독'),('fixed_ensemble','고정 첫seed 치환')]:
  q=summaries[n]['comparisons'][key];lo,hi=q['conditional_CI95'];rows.append(f"| {n} {label} | {scope} | {q['baseline_RMSE']:.6f}→{q['candidate_RMSE']:.6f} | {q['gain_baseline_minus_candidate']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {q['top5_removed_gain']:+.6f} | {q['positive_folds']}/5 |")
old=read(_repo_path(P, 'reports/cnn-controls-summary.json'));score={v['run']:v['scores']['cnn']['oof_rmse'] for v in old};p128=read(_repo_path(P, 'artifacts/run-009/plan.json'));p256=read(_repo_path(P, 'artifacts/run-010/plan.json'));assert (p128['image_size'],p128['epochs'],p256['image_size'],p256['epochs'])==(128,80,256,40)
partials=[]
for line in (_repo_path(P, 'docs/records/roadmap-audit/roadmap-30-after-215.md')).read_text().splitlines():
 if re.match(r'\| \d{2}\.',line) and line.split('|')[2].strip()=='부분':partials.append(line)
assert len(partials)==8
sources=['reports/cnn-controls-summary.json','artifacts/run-009/plan.json','artifacts/run-010/plan.json','reports/cnn-resolution-188-summary.json','docs/records/roadmap-audit/roadmap-30-after-215.md','docs/records/roadmap-audit/215-management.json']
for n,name in names.items():sources.extend([f'experiments/{name}/REPORT.md',f'experiments/{name}/plan.json',f'experiments/{name}/final-audit.json',f'experiments/{name}/summary.json'])
write(_repo_path(H, 'source-evidence.json'),{n:sha(_repo_path(P, n)) for n in sources});write(_repo_path(H, '213-215-summary.json'),{'experiments':summaries,'actual_new_training_runs':21,'candidate_runs':15,'verification_runs':6,'historical_controls':'the same five206 controls reused per experiment;not15new fits','selection_bias':'same400 reused;CIconditionalonfits','best021_preserved':True,'no_automatic_followup':True})
text=f"""# 215 완료 후 인계

**213–215의 첫 단일 비교를 모두 완료했다. best021을 유지하고 학습은 종료했다. 새 실험을 예약하거나 추가 강도·재분할·후단 재학습을 시작하지 않았다.** 최신 [30항목 관리표](../roadmap-30-after-215.md)의 범위는 **완료22·부분8·미실시0**이다. 완료는 수행 범위이며 성능 문제 해결이나 기법 전체 검증을 뜻하지 않는다.

## 바로 필요한 결과

이득은 대조RMSE−후보RMSE다. 같은 개발400의 원본5fold이며 모든CI는 고정된fit에 조건부다.

| 실험 | 평가범위 | RMSE 대조→후보 | 이득 | paired95%CI | 상위5개 제외 이득 | 개선fold |
|---|---|---|---:|---|---:|---:|
"""+'\n'.join(rows)+f"""

각 실험의 사전 기준과 판단은 원본 보고서에 보존했다: [213 밝기](../../experiments/brightness-augmentation-213/REPORT.md), [214 감마](../../experiments/gamma-augmentation-214/REPORT.md), [215 블러](../../experiments/blur-augmentation-215/REPORT.md). 상세수치는 [구조화 요약](213-215-summary.json)에 있다. 단독 이득과 고정 첫seed치환을 구별하며 약한 치환이득을 전체모델 개선으로 취급하지 않는다. **213/214 재분할은 하지 않는다는 최신 지시를 유지한다.** 215도 이번 한정책 비교에서 정지한다.

실제 새학습은 각7회, 합계 **21회/16,800update**다. 후보15회＋재현검증6회이며 이전206의 동일한대조5개를 각실험에서 재사용했다. 재사용을 새15회 학습으로 세지 않는다. 분할·seed·D4·MSE·optimizer·80epoch·마지막5state평균 고정, 대조exact재현·10모델재로드·라벨NaN교란·bootstrap10,000회 독립재검산을 각실험에서 통과했다. 원본과best021 SHA256 `7cf4ce3bbef6b98d5a06e5b3b19498379ca5fe4f076bbdc95d9afa55cc893a4b` 보존.

반복 사용한 개발400은 독립 검증이 아니다. CI는 누적선택편향·다른seed·재학습 불확실성을 보정하지 않는다. 동일400에서 서로다른정책의 점수를 비교해213을 승자로 선택하거나 가중치를 다시 최적화하지 않았다.

## 세 변환에서 확인한 사실

- **213 밝기:** train전체 고정정규화이므로 밝기이동이 상쇄되지 않는다. 206 평균보존대비감소와 다르다. train평균 IQR기반 작은덧셈,40원본/20밝게/20어둡게. 클리핑·재양자화없음. 단독약한신호를 최종앙상블개선으로 승격하지 않는다.
- **214 감마:** train-only 약0.985–1.015의비선형단조변환,50%노출. 순위가보존돼도 fixedthreshold0.5의어두운분율이 model128에서최대3.17%p,trainpixelQ10기준5.84%p변했다. 모든구조측정보존 주장은불가하다.
- **215 블러:** train선명도통계로3tap `[a,1−2a,a]` 각축, a약0.0373–0.0391,reflect/50%노출. 원본·입력해상도와좌표는유지되지만 model128 gradient energy는원본의68–85%,Laplacian분산은63–76%로감소했다. threshold0.5에서원래1–4픽셀영역의완전소실이최대166개/영상이었다. 사전선정train12장/fold의기존sigma1 watershed 영역수변화는model128 −66~+5였다. 이들은작은공간필터도측정에영향을준다는증거이며실제상·결정립정답변화를측정한것은아니다. 기존097/098/183의측정타당성문제는해결되지않았다.

## 로드맵의 남은 부분 항목 8개

| 번호 | 남은 점 | 후속 판단 |
|---|---|---|
| 01 Watershed 크기/d^−1/2 | 구현과시험은했으나물리정답·길이스케일부재,분할안정성부족 | 새독립정답없이고강도분할·평활화·183역렌더링을반복하지않음 |
| 02 어두운상/기공 | threshold·Hessian은시험했으나실제상/경계/기공분리미확정 | 214/215에서측정이변함을보였을뿐진짜분율검증아님 |
| 09 ConvNeXt | 도입/pooling/head/laststage비교완료;동일용량중간stage정보대조미확인 | 190–192/210실패정책을층/head탐색으로자동확장하지않음 |
| 12 원해상도256 | Tiny128/80 대Tiny256/40으로예산혼재;ResNet188은224→256비교 | 아래단일우선가설로동일업데이트비교제안만 |
| 14 LLRD | 마지막stage내부고정ratio0.8종료;전체backbone범위는미검증 | 미세조정층·비율확대비용대비근거약함 |
| 21 Noise증강 | 110동결DINO앞Gaussian노이즈는완료;trainableTinyCNN의학습중노이즈는미확인 | 미실행공백만으로추가하지않음;213–215전체조합탐색도하지않음 |
| 23 타깃층화 | 같은400의GroupKFold반복은완료;타깃층화미실시 | 평가균형을위한설계이지모델자체성능향상또는독립검증이아님 |
| 30 코드/가중치라이선스 | 사용파일별기록은있으나MicroNet버전/개별가중치권리불확실성일부 | 모델개선실험과분리;이번인계에서외부조회·다운로드없음 |

## 다음 우선 가설 한 개

**현재 TinyCNN에서128→원본256 입력정책을 같은80epoch/800update로 비교하면, 기존BOX축소에서 사라지는 미세경계정보를 이용해 예측오차를 실질적으로 낮출 수 있는가.** 남은항목중 실행가능성과프로젝트내기존신호를근거로우선순위를부여한판단이며큰개선의확률이나규모를입증한것은아니다. 실험번호·실행은예약하지않는다.

근거는 [009/010 원기록](../../reports/cnn-controls-summary.json)에서 Tiny128/80 RMSE **{score['run-009']:.6f}**, Tiny256/40 **{score['run-010']:.6f}**였다는점이다. 하지만 epoch와cosine스케줄길이도달라해상도의순수이득이라고말할수없다. [009 계획](../../artifacts/run-009/plan.json)과 [010 계획](../../artifacts/run-010/plan.json)을실제로확인했다. 다른모델인 [188 ResNet224→256](../../reports/cnn-resolution-188-summary.json)은단독+0.113239,고정결합+0.011518이지만CI가0을포함하므로보조적이고약한근거다. 215의경계·작은영역민감도는입력상세정보가영향을줄수있다는진단이지,더높은해상도가회귀를개선한다는증명은아니다.

실행하게된다면한쌍만사전고정한다:현재128/80대256/80,같은원본5fold·seed·모델·batch32·D4·MSE·AdamW·80epoch/800update·마지막5state평균. 새로운증강·seed·해상도격자를추가하지않는다. train-only정규화는각입력경로에서계산하고원본train400만사용한다. 먼저기존환경에서6GB이내의실행가능성을확인한다. 이는같은업데이트예산이지동일FLOPs/시간예산은아니며,동일CNN의물리적receptivefield도입력해상도에따라달라지는**입력정책대조**다.

우선판정은CNN단독의pairedCI·fold방향·상위5개제외를동일기준으로보고,고정첫seed치환을별도진단으로둔다. 좋은단독점수또는작은치환이득만으로best021을교체하거나전체모델개선을주장하지않는다. 실제최종파이프라인개선을확정하려면별도단계의독립성/안정성확인과일관된전체후단평가가필요하지만,여기서204의대규모nested분기를재시작하거나전체500/test/제출로넘어가지않는다. 기준미달이면다른해상도·epoch로구제탐색하지않는다.

이하항목보다우선한이유:검증된물리정답확보는현재허용자료로실행할수없고,추가ConvNeXt/LLRD는종료된실패경로의확대이며,noise는실행공백외에큰실질이득의근거가약하고,타깃층화/라이선스는평가·관리개선이다. 이판단역시기존자료에조건부이며새로운독립데이터가생기면우선순위가달라질수있다.

## 재개에 필요한 경로와 경계

- 환경: 프로젝트 `.venv/bin/python -I -B`, CPU2threads/MPS,메모리6GB미만. 213–215설치·다운로드·비용0.
- 보고서/plan/source/state/ID잔차는 `experiments/{{brightness-augmentation-213,gamma-augmentation-214,blur-augmentation-215}}/`에보존. 기술근거는각 final-audit.json,validation.json,preflight.json,code-lock.json과fold별streams다.
- latest관리: [215 JSON](../215-management.json),[30항목](../roadmap-30-after-215.md). 근거hash는 [source-evidence.json](source-evidence.json).
- 기존완료폴더의run_stage.py를재실행하지않는다. 출력은배타적생성이며학습/검증script는읽기전용이아니다. 새실행에는새출력경로·고정계획이필요하다.
- 원본/best보존,설치·다운로드·회사자료·Slack·인증정보·test·이전holdout·전체500·CSV생성/수정·제출금지조건을유지한다. 비용0. 보고서·인계완료시점실행중학습없음. 차단사항없음.
"""
with (_repo_path(H, 'HANDOFF.md')).open('x') as f:f.write(text)
for target in re.findall(r'\]\(([^)]+)\)',text):
 if '://' not in target:assert (_repo_path(H, target.split('#')[0])).exists(),target
write(_repo_path(R, 'handoff-delivery.json'),{'passed':True,'handoff':str((_repo_path(H, 'HANDOFF.md')).relative_to(P)),'files':{str(p.relative_to(P)):sha(p) for p in H.iterdir() if p.is_file()},'new_training_runs_213_to215':21,'remaining_partial_count':8,'next_hypothesis_count':1,'executed_next_hypothesis':False});print(json.dumps({'handoff_written':str(_repo_path(H, 'HANDOFF.md')),'next_hypothesis':'TinyCNN128vs256same80epoch','automatic_followup':False},ensure_ascii=False))
