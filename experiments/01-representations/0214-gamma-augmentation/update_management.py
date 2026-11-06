from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,datetime
P=_RepoPath(__file__).resolve().parents[3];R=_repo_path(P, 'experiments/gamma-augmentation-214')
def read(p):return json.loads(p.read_text())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
s=read(_repo_path(R, 'summary.json'));done=read(_repo_path(R, 'completion.json'));audit=read(_repo_path(R, 'final-audit.json'));assert audit['passed'] and done['completed']
a=s['comparisons']['standalone'];b=s['comparisons']['fixed_ensemble']
mg={'study':214,'status':'완료','decision':done['decision'],'scope_counts':{'완료':21,'부분':8,'미실시':1},'changed_rows':{'17':'211손실대조/212재분할 완료·미채택 유지','18':'213밝기 비교 완료·CI기준 미달·best021유지;명시지시에 따라 재분할 안 함','20':'214고정 감마 비교 완료·현재 결과는 보고서 참조','22':'Blur 미실시·자동 실행 없음'},'CNN_standalone':a,'fixed_first_seed_replacement':b,'historical_213_decision':'no additional repartition','EMA209_status':'보류','nested204_retry':False,'new_training_running':False,'new_training_runs':7,'best021_maintained':True,'report':'experiments/gamma-augmentation-214/REPORT.md','latest_roadmap':'docs/records/roadmap-audit/roadmap-30-after-214.md','original_snapshots_preserved':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
write(_repo_path(P, 'docs/records/roadmap-audit/214-management.json'),mg)
source=(_repo_path(P, 'docs/records/roadmap-audit/handoff-after-210/roadmap-30.md')).read_text().split('## 다음 세션의 잔여 검토: 최대 두 개')[0]
source=source.replace('# 210 이후 로드맵 30개 항목 관리표','# 214 이후 로드맵 30개 항목 관리표').replace('**완료 18개 · 부분 9개 · 미실시 3개**','**완료 21개 · 부분 8개 · 미실시 1개**').replace('205–210을 반영했다.','205–214를 반영했다. 과거 스냅샷은 보존했다.')
lines=[]
for line in source.splitlines():
 if line.startswith('| 17.'):
  line='| 17. Huber loss | 완료 | **211** 원본 단독 +0.439337이나 CI0포함. **212** 두 재분할 단독 pooled −0.176793, 고정 치환 −0.024428. [212](../experiments/huber-repartition-212/REPORT.md) | 현재 TinyCNN MSE–2×Huber(delta1) 고정 대조 완료·미채택. 큰 오차를 라벨오류로 단정하지 않음. |'
 elif line.startswith('| 18.'):
  line='| 18. 밝기 단일 증강 | 완료 | **213** float 덧셈 밝기 단독 +0.197015, CI[−0.035373,+0.434660]. 고정 치환 +0.007204, 상위5개 제외 −0.003433. [213](../experiments/brightness-augmentation-213/REPORT.md) | 사전기준 미달·best021유지. 최신 지시에 따라 재분할하지 않음. 고정train정규화에서 밝기가 남음을 수치 확인. |'
 elif line.startswith('| 20.'):
  line=f"| 20. 감마 단일 증강 | 완료 | **214** train-only 작은 log-gamma 한 정책, 단독 {a['gain_baseline_minus_candidate']:+.6f}, CI{a['conditional_CI95']}; 고정 치환 {b['gain_baseline_minus_candidate']:+.6f}, 상위5개 제외 {b['top5_removed_gain']:+.6f}. [214](../experiments/gamma-augmentation-214/REPORT.md) | {done['decision']} 고정threshold dark fraction 변화 확인; 단조성이 모든 구조측정 보존을 뜻하지 않음. 재분할·강도탐색 없음. |"
 # Existing links in the archived roadmap were two levels deep; this snapshot lives directly in roadmap-audit.
 if not line.startswith(('| 17.','| 18.','| 20.')):line=line.replace('../../','../').replace('(../audit.md)','(audit.md)').replace('(../209-status-after-review.md)','(209-status-after-review.md)').replace('(../verification.json)','(verification.json)')
 lines.append(line)
text='\n'.join(lines)+"\n## 최신 실행 경계\n\n213 밝기 재분할은 진행하지 않는다. 214 감마 단일 정책 비교를 완료했고 추가 강도나 재분할은 자동 진행하지 않는다. Blur는 유일한 미실시 항목으로 남으며 별도 지시 전 실행 대기열로 만들지 않는다. 완료는 수행 범위이지 성능 향상이나 물리적 라벨 불변성의 증명이 아니다. 같은 개발400의 반복 선택 편향은 조건부CI에 반영되지 않았다. best021과 기존 원본·과거관리표를 보존한다.\n"
rows=[line for line in lines if __import__('re').match(r'\| \d{2}\.',line)];assert len(rows)==30
counts={k:sum(line.split('|')[2].strip()==k for line in rows) for k in ['완료','부분','미실시']};assert counts==mg['scope_counts'],counts
with (_repo_path(P, 'docs/records/roadmap-audit/roadmap-30-after-214.md')).open('x') as f:f.write(text)
with (_repo_path(P, 'docs/records/roadmap-audit/214-completion.md')).open('x') as f:f.write(f"# 214 완료\n\n{done['decision']}\n\n실제학습7회(후보5+재현검증2), 대조5재사용. 단독이득 {a['gain_baseline_minus_candidate']:+.8f}, 고정치환이득 {b['gain_baseline_minus_candidate']:+.8f}. [보고서](../experiments/gamma-augmentation-214/REPORT.md), [최신30항목](roadmap-30-after-214.md), [관리JSON](214-management.json).\n\n밝기213재분할 없음. 감마214다른강도/재분할 없음. Blur미실시. 원본과best021보존. 완료21·부분8·미실시1.\n")
write(_repo_path(P, 'reports/gamma-augmentation-214-summary.json'),s)
outs=[_repo_path(P, 'docs/records/roadmap-audit/214-management.json'),_repo_path(P, 'docs/records/roadmap-audit/214-completion.md'),_repo_path(P, 'docs/records/roadmap-audit/roadmap-30-after-214.md'),_repo_path(P, 'reports/gamma-augmentation-214-summary.json')];write(_repo_path(R, 'management-delivery.json'),{'passed':True,'verified_30_row_counts':counts,'files':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outs}})
print(json.dumps({'management_updated':True,'scope_counts':counts,'files':[str(p.relative_to(P)) for p in outs]},ensure_ascii=False))
