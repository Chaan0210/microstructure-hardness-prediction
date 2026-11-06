from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,csv,hashlib,datetime,io
P=_RepoPath(__file__).resolve().parents[3];R=Path(__file__).resolve().parent
assert P==Path(str(_REPO_ROOT))
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,z):
 with p.open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False,allow_nan=False)
plan=read(_repo_path(R, 'plan.json'));protected=read(_repo_path(R, 'preservation.json'));lock=read(_repo_path(R, 'code-lock.json'));now=datetime.datetime.now(datetime.timezone.utc).isoformat()
for n,h in protected.items():assert sha(_repo_path(P, n))==h,n
for n,h in plan['inputs_SHA256'].items():assert sha(_repo_path(P, n))==h,n
for n,h in lock.items():assert sha(_repo_path(R, n))==h,n
exitcodes={n:read(_repo_path(R, n+'-exit.json'))['exit_code'] for n in ['baseline','study','validate','score','score_validate']};assert set(exitcodes.values())=={0}
null=read(_repo_path(R, 'baseline-validation.json'));valid=read(_repo_path(R, 'validation.json'));sv=read(_repo_path(R, 'score-validation.json'));report=read(_repo_path(R, 'report.json'));assert null['passed'] and valid['passed'] and sv['passed'] and valid['new_fit_count']==30
assert len(list(R.glob('20*/fold-*/model.pt')))==30
trace={n:read(_repo_path(R, n+'-resources.json')) for n in ['baseline','study','validation']};peak=max(max(t['RSS'],t['peak_RSS'])+t['MPS'] for r in trace.values() for t in r['trace']);assert peak<6e9 and all(max(t['swap'] for t in r['trace'])==r['initial_swap'] for r in trace.values());resources={'CPU_threads':2,'device':'MPS float32; CPU reload verification','peak_RSS_plus_MPS':peak,'swap_growth':0,'monitored_seconds_by_phase':{n:r['trace'][-1]['seconds'] for n,r in trace.items()},'monitored_total_seconds':sum(r['trace'][-1]['seconds'] for r in trace.values()),'timing_scope':'training/nullcertification and allmodelvalidation windows; excludes implementation/imports/inputhashes/scoring/reporting','fits':30,'optimizer_updates':24000,'extra_validation_fits':0,'cost':0}
resources['sum_fit_seconds']=sum(read(q)['seconds'] for q in R.glob('20*/fold-*/report.json'))
closed=not report['all_new_partitions_passed'];action='고정 CNN 보조 CE 분기 종료; best021 유지' if closed else '제한적 재현 결과 보고; best021 유지; 추가 실행 없음'
summary={'study':201,'status':'completed_branch_closed_keep021' if closed else 'completed_report_only_keep021','UTC':now,'purpose':'193 효과의 기존 두 개발 재분할 안정성 확인 및 분기 판단.022 Public 구제/새후보 선택이 아니다.','decision':report['decision'],'conclusion':action,'all30_training_and_validation_completed':True,'all10_detach_exact_prior_weights_and_OOF':True,'individual_partitions':report['individual_partitions'],'IDpaired_three_partition_diagnostic':report['IDpaired_three_partition_diagnostic'],'historical194_original_CI95':[-0.008093455309132214,0.058825509600061394],'bootstrap_note':'201은 원본/두재분할 모두 고정 seed202710201의 공통3000draw로 계산.194의 원본3000draw seed202710194 역사적 CI와 구분.400ID를 재표집하며 각ID의3분할 측정을 묶음; 독립1200으로 취급하지 않음.','original194_gate_remains_failed':True,'recipe':plan['recipe'],'fixed_replacement':plan['frozen_replacement'],'calibrators':plan['calibrators'],'canonical_precision':plan['precision'],'validation':{k:v for k,v in valid.items() if k!='checks'},'score_validation':sv,'resources':resources,'limits':plan['limits'],'current_best':'021','best021_SHA256':sha(_repo_path(P, 'artifacts/submission-candidate-021.csv')),'candidate022_SHA256':sha(_repo_path(P, 'artifacts/submission-candidate-022.csv')),'new_holdout_test_full500_access':False,'new_submission_CSV_or_upload':False,'new_install_download_cost':False,'coefficient_seed_epoch_weight_search':False,'code_lock_SHA256':sha(_repo_path(R, 'code-lock.json')),'plan_SHA256':sha(_repo_path(R, 'plan.json')),'report_SHA256':sha(_repo_path(R, 'report.json')),'OOF_SHA256':{s:sha(_repo_path(R, s, 'OOF.npz')) for s in ['20261031','20261101']},'remaining_blockers':[]}
summary['interpretation']={
 'observed_signal':'True CE는 detach에 대한 단독과 고정치환 RMSE를 세분할 모두 개선했다.효과가 전혀 없다고 결론내리지 않는다.',
 '20261031_failure':'고정치환 gain+.00838142로 최소.02미달,95%CI0포함,상위5이득제외−.00923673.순열대조에는낫지만best021대비안정성미달.',
 '20261101_failure':'best021대비gain+.03620072와CI하한양수/4fold/상위5제외통과.그러나순열CE가단독14.16668915로true14.26330674보다좋고,치환true대순열은2/5fold·상위5제외−.01442661로paired기준실패.',
 'aggregate':'400ID를묶은3분할오차제곱평균의true대021gain+.02323278,95%CI[.00527782,.04042370].true대순열gain+.01605850의CI[−.00488440,.03639214]는0포함.종합양수CI가개별분할실패나CE정답정보특이성부족을대체하지않는다.',
 'action':'좋은분할만선택하거나계수/seed/epoch로구제하지않고현재고정CNNCE분기를종료한다.다른모든CE설계에대한보편적기각이나독립일반화결론은아니다.'}
write(_repo_path(R, 'summary.json'),summary);write(_repo_path(P, 'reports/auxiliary-repartition-201-summary.json'),summary)
review={'range':'193–201','UTC':now,'current_best':{'candidate':'021','Public_RMSE_parent_reported':11.4878517800,'CSV_SHA256':summary['best021_SHA256']},'conclusions':[
 {'study':'193','result':'CNN trueCE가 원본 단독에서는 detach/순열보다 개선됐지만 고정10%추가 RMSE12.79322535는 best02112.78620078보다 나쁨.','status':'추가 방식 실패; 원본 효과의 재분할 검증은201에서 완료','evidence':'reports/auxiliary-cnn-193-summary.json'},
 {'study':'194','result':'기존 첫CNN seed만 고정출력 치환: 원본12.78620078→12.76150457, gain.02469621.역사적95%CI[-.00809346,.05882551]로 채택기준 미달.','status':'194실패 유지;201이 두재분할 안정성 판단을 추가','evidence':'experiments/matched-replacement-194/summary.json'},
 {'study':'195','result':'기존DINO6912D에 최종max/min각768추가, 동일EDF32에서 양쪽모두 단독·결합 악화.','status':'분기 종료; signedGeM은 별개 미실행','evidence':'experiments/pooling-extrema-195/summary.json'},
 {'study':'196–197','result':'crossQ직전 LN으로 MHA단독16.44024945→15.82276548 개선.고정10%결합12.82464418은021보다 나쁨.동일파라미터수는 동일유효용량을 보장하지 않음.','status':'사전 채택기준 미달; 가중치/온도/위치/구조 탐색 없음','evidence':'experiments/query-normalization-197/summary.json'},
 {'study':'198','result':'명시 탐색선택 후보022를 전체500로 만들고 검증 후 제출.실제Public11.5435627849로021보다.0557110049 악화.동일데이터/seed true반복exact.','status':'기록보존·021유지; 기술검증 통과는 일반화 보증이 아님','evidence':'reports/candidate022-198-summary.json'},
 {'study':'199','result':'원본194의 작은 이득이 상위5표본에73.1%집중, 상위10이득제외 시악화.동일400의OOF/full500학습내delta상관.02460.새기술오류 증거는 찾지못했으며 원인확정은 못함.','status':'사후진단 종료; 제안두개는200/201로 모두완료','evidence':'reports/submission022-diagnostic-199-summary.json'},
 {'study':'200','result':'197의cross균등화 예측deltaRMS9.21465 vs196.11613: 학습된 비균등경로에 실제 의존.197단독은15.82277→17.13870악화.고정10%결합12.78588580의021대비.00031497 이득도 후보로선택하지않음.','status':'가중치고정 경로진단 종료·새학습/모델선택 없음','evidence':'reports/attention-counterfactual-200-summary.json'},
 {'study':'201','result':{s:{'best021':q['scores']['best021']['RMSE'],'true':q['scores']['true']['RMSE'],'permuted':q['scores']['permuted']['RMSE'],'true_gain':q['pairs']['true_vs_best021']['gain'],'paired_passed':q['auxiliary_paired_stability_passed'],'reference_passed':q['reference_194criteria_passed']} for s,q in report['individual_partitions'].items()},'status':action,'evidence':'reports/auxiliary-repartition-201-summary.json'}],
 '051_related_failure':'FrozenDINO sharedMLP CE의 원본결합gain+.02041가 두재분할−.04928/−.00258로 불안정했다.193은CNN학습층/4bin/MSE/분리clip대조가 달라051의 직접재실험이나 동일모델은 아니다.201을 함께 보고 판단한다.',
 'remaining_unexecuted':[{'item':'signedGeM','status':'음수token에 대한 signed정의·집계식 미확정; 구현/실행 없음.195max/min 결과로 모든pooling을 일반기각하지 않음.'},{'item':'보조CE 추가계수/seed/epoch/3seed전체교체/downstream재학습','status':'미실시.현재 고정 CNN CE 분기 종료 후 구조조정·구제탐색으로 이어가지 않음.' if closed else '미실시.이번30fits 범위밖이며 자동확대 없음.'},{'item':'200 균등화조건 후보승격/가중치탐색','status':'미실시·분기종료.기존 비선택 결정 유지.'},{'item':'201 전체500/test/새제출CSV/업로드','status':'미실시.재분할 연구만 완료; best021 유지.'},{'item':'외부DSB2018 일부세부주장','status':'이전 부모전달 조사만 기록;본문독립확인 미완료.현성과근거로 추가사용하지 않음.'}],
 'completed_not_pending':['193의 기존 두 개발 재분할30fits:201완료','199의 frozen cross균등개입 제안:200완료','전체500 후보022 생성·검증·부모제출:198/199기록완료'],
 'uncertainty':'193–201은 동일 개발400의 반복연구가 많다.201세분할은 같은400ID이며 독립1200이 아니다.모든조건부CI는학습재추정과누적선택불확실성을빠뜨린다.작은원본이득/재현성/기술정확성만으로 Public/Private일반화를 보증하지 않는다.',
 'current_task_blockers':[]}
write(_repo_path(R, 'research-193-201-summary.json'),review);write(_repo_path(P, 'reports/research-193-201-summary.json'),review)
reg=_repo_path(P, 'reports/strategy-experiment-register.csv');before=reg.read_bytes();assert hashlib.sha256(before).hexdigest()==protected[str(reg.relative_to(P))]
with (_repo_path(R, 'strategy-experiment-register.before.csv')).open('xb') as f:f.write(before)
reader=csv.DictReader(io.StringIO(before.decode()));fields=reader.fieldnames;rows=list(reader);oldrows=json.loads(json.dumps(rows));assert not any(q['실험'].startswith('201') for q in rows)
for q in rows:
 n=q['실험']
 if n=='193':q['다음행동']='194 고정첫seed치환 및201 두재분할30fits 완료. '+action;q['갱신UTC']=now
 elif n=='194':q['다음행동']='194 CI기준미달 유지.198후보022 실제Public악화 기록보존.201 두재분할고정치환 완료. '+action;q['갱신UTC']=now
 elif n.startswith('199 '):q['다음행동']='제안 두개 모두 완료:200 저장cross균등진단(학습0,후보비선택),201 기존두재분할30fits. '+action;q['갱신UTC']=now
 elif n.startswith('200 '):q['다음행동']='200 균등조건 .00031497 우위는 선택하지않고종료.후속명시승인으로201 두재분할30fits완료. '+action;q['갱신UTC']=now
for old,new in zip(oldrows,rows):
 changed=[k for k in old if old[k]!=new[k]];assert set(changed)<= {'다음행동','갱신UTC'}
result_text='; '.join(f"{s}: 021 {q['scores']['best021']['RMSE']:.10f} / true {q['scores']['true']['RMSE']:.10f} / permuted {q['scores']['permuted']['RMSE']:.10f}" for s,q in report['individual_partitions'].items())
row={'실험':'201 / CNN CE 재분할안정성','출처':'193/194 후속 및 부모명시승인;199제안1','가설':'193의 CNN trainquartile CE 효과와194고정첫seed치환이 기존두재분할에서도 안정적인가','기존 중복과 차이':'원본400대신 동일400의기존두재분할.학습recipe/seed/.3CE/80epochs와split별기존보정계수고정.022구제아님','대조군':'각분할 true/detach/permuted x5fold=30fits.10detach 기존가중치/OOF exact 먼저확인','사전고정조건':'CPU2/MPS;193학습루프동일;80epochs800updates/last5평균;trainonlynorm/quartile;Q=P021+.081*s_fold*(newfirstCNN-oldfirstCNN)','채택·기각기준':'두대조대비원본단독gain및고정치환gain/3fold/top5제외;021gain>=.02/3fold/조건부CI하한>0/top5제외.어느재분할실패시분기종료,가중치/seed/epoch탐색없음','상태':'완료·'+action,'결과와 당시 기준·split':result_text,'해석·검증 한계':'같은400ID의3반복측정.400ID paired bootstrap3000공통draw,독립1200CI아님.10nullfullreplays exact;30CPU/MPSreload와trainonlypoison통계/optimizer/D4/LR/5state평균검증.조건부CI/누적선택편향미보정','다음행동':action+'; 전체500/test/CSV/업로드 없음.193–201종합기록완료','근거파일':'experiments/auxiliary-repartition-201/plan.json;summary.json;baseline-validation.json;validation.json;score-validation.json;completion.json;reports/research-193-201-summary.json','갱신UTC':now}
assert set(row)==set(fields);out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows([row,*rows]);reg.write_bytes(out.getvalue().encode())
assert len(list(csv.DictReader(io.StringIO(reg.read_text()))))==len(oldrows)+1
unchanged=0
for n,h in protected.items():
 if n!=str(reg.relative_to(P)):assert sha(_repo_path(P, n))==h,n;unchanged+=1
for n,h in plan['inputs_SHA256'].items():assert sha(_repo_path(P, n))==h,n
for n,h in lock.items():assert sha(_repo_path(R, n))==h,n
files={}
for q in sorted(R.rglob('*')):
 if q.is_file() and not any(n in q.relative_to(R).parts for n in ['cache','config','tmp','__pycache__']):files[str(q.relative_to(R))]={'bytes':q.stat().st_size,'SHA256':sha(q)}
write(_repo_path(R, 'manifest.json'),{'files':files,'scope':'201 artifacts before manifest/completion; ownhash in completion; external report copies listed there'})
completion={'study':201,'status':summary['status'],'UTC':now,'all_stage_exit_codes':exitcodes,'new_fullfits':30,'extra_fits':0,'new_submission':False,'best':'021','protected_unchanged_except_authorized_registry':unchanged,'registry_before_SHA256':hashlib.sha256(before).hexdigest(),'registry_after_SHA256':sha(reg),'summary_SHA256':sha(_repo_path(R, 'summary.json')),'external_summary_SHA256':sha(_repo_path(P, 'reports/auxiliary-repartition-201-summary.json')),'research193_201_SHA256':sha(_repo_path(R, 'research-193-201-summary.json')),'external_research193_201_SHA256':sha(_repo_path(P, 'reports/research-193-201-summary.json')),'manifest_SHA256':sha(_repo_path(R, 'manifest.json')),'finalizer_SHA256':sha(_repo_path(R, 'finalize.py')),'best021_SHA256':summary['best021_SHA256'],'candidate022_SHA256':summary['candidate022_SHA256'],'resources':resources,'remaining_blockers':[]}
write(_repo_path(R, 'completion.json'),completion);print(json.dumps(completion,ensure_ascii=False));print('completion_SHA256',sha(_repo_path(R, 'completion.json')))
