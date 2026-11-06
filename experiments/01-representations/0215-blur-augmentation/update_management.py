from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,re,datetime
P=_RepoPath(__file__).resolve().parents[3];R=_repo_path(P, 'experiments/blur-augmentation-215')
def read(p):return json.loads(p.read_text())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
s=read(_repo_path(R, 'summary.json'));done=read(_repo_path(R, 'completion.json'));audit=read(_repo_path(R, 'final-audit.json'));assert audit['passed'] and done['completed'];a=s['comparisons']['standalone'];b=s['comparisons']['fixed_ensemble']
mg={'study':215,'status':'완료','decision':done['decision'],'scope_counts':{'완료':22,'부분':8,'미실시':0},'remaining_partial_rows':[1,2,9,12,14,21,23,30],'CNN_standalone':a,'fixed_first_seed_replacement':b,'new_training_runs':7,'best021_maintained':True,'report':'experiments/blur-augmentation-215/REPORT.md','handoff':'docs/records/roadmap-audit/handoff-after-215/HANDOFF.md','new_training_running':False,'automatic_resplit_or_strength_search':False,'prior_snapshots_preserved':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
write(_repo_path(P, 'docs/records/roadmap-audit/215-management.json'),mg)
source=(_repo_path(P, 'docs/records/roadmap-audit/roadmap-30-after-214.md')).read_text().split('## 최신 실행 경계')[0].replace('# 214 이후','# 215 이후').replace('완료 21개 · 부분 8개 · 미실시 1개','완료 22개 · 부분 8개 · 미실시 0개').replace('205–214','205–215');lines=[]
for line in source.splitlines():
 if line.startswith('| 22.'):
  line=f"| 22. Blur 단일 증강 | 완료 | **215** 작은3tap blur/50%노출 한 정책. 단독 {a['gain_baseline_minus_candidate']:+.6f}, CI{a['conditional_CI95']}, 상위5제외 {a['top5_removed_gain']:+.6f}; 고정치환 {b['gain_baseline_minus_candidate']:+.6f}, 상위5제외 {b['top5_removed_gain']:+.6f}. [215](../experiments/blur-augmentation-215/REPORT.md) | {done['decision']} 선명도·작은상·watershed대리측정 변화 확인;라벨보존 미입증. 자동재분할/강도탐색 없음. |"
 if line.startswith('| 12.'):line=line.replace('지금 해상도 탐색을 반복하지 않음','215 이후 인계에서 동일80epoch 대조를 다음 우선 가설로 검토하되 자동실행하지 않음')
 lines.append(line)
text='\n'.join(lines)+"\n## 최신 실행 경계\n\n213–215의 밝기·감마·블러 한 정책씩 완료했다. 첫seed치환 효과를 전체후단 재학습 효과로 취급하지 않는다. 213/214재분할과 각강도탐색을 생략했고215도 비교 후 정지한다. 다음 우선가설1개와 남은부분8개는 [HANDOFF](handoff-after-215/HANDOFF.md)에 정리한다. 수행완료는 문제해결이나물리라벨보존의 증명이 아니다. 원본·best021·과거관리표를 보존한다.\n"
rows=[l for l in lines if re.match(r'\| \d{2}\.',l)];counts={k:sum(l.split('|')[2].strip()==k for l in rows) for k in ['완료','부분','미실시']};assert len(rows)==30 and counts==mg['scope_counts']
with (_repo_path(P, 'docs/records/roadmap-audit/roadmap-30-after-215.md')).open('x') as f:f.write(text)
with (_repo_path(P, 'docs/records/roadmap-audit/215-completion.md')).open('x') as f:f.write(f"# 215 완료\n\n{done['decision']}\n\n새학습7회(후보5+검증2), 대조5재사용. 완료22·부분8·미실시0. [보고서](../experiments/blur-augmentation-215/REPORT.md) · [인계](handoff-after-215/HANDOFF.md) · [최신관리표](roadmap-30-after-215.md).\n")
write(_repo_path(P, 'reports/blur-augmentation-215-summary.json'),s)
outs=[_repo_path(P, 'docs/records/roadmap-audit/215-management.json'),_repo_path(P, 'docs/records/roadmap-audit/215-completion.md'),_repo_path(P, 'docs/records/roadmap-audit/roadmap-30-after-215.md'),_repo_path(P, 'reports/blur-augmentation-215-summary.json')];write(_repo_path(R, 'management-delivery.json'),{'passed':True,'verified_30_row_counts':counts,'files':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outs}});print(json.dumps({'management_updated':True,'scope_counts':counts},ensure_ascii=False))
