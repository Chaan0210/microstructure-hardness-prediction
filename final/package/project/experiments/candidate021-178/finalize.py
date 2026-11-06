"""Collect verified178 evidence without changing fixed models or predictions."""
import csv,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parents[2];R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import runtime as f

def load(p):return json.loads(p.read_text())
def main():
 plan=f.check();v=load(R/'final-inference/validation.json');v2=load(R/'reproduction-check/validation.json');assert v['passed'] and v2['passed']
 csvpath=P/'artifacts/submission-candidate-021.csv';hh=f.sha(csvpath)
 assert hh==v['CSV_sha256']==v2['CSV_sha256']==f.sha(R/'final-inference/submission-candidate-021.csv')==f.sha(R/'reproduction-check/submission-candidate-021.csv')
 with csvpath.open(newline='') as fp:
  rd=csv.DictReader(fp);assert rd.fieldnames==['ID','hardness'];rows=list(rd)
 assert [x['ID'] for x in rows]==[f'TEST_{i:06d}' for i in range(1,1001)]
 import math
 assert all(math.isfinite(float(x['hardness'])) for x in rows)
 audit=load(R/'selection-audit.json');train=load(R/'training-validation.json');assert train['passed'];feat={}
 for split in ['train','test']:
  a=load(R/f'{split}-feature-validation.json');assert a['passed'];feat[split]={k:x for k,x in a.items() if k!='image_sha256'}
 resources={}
 for file in sorted(R.glob('*-resources.json')):
  a=load(file);z=a['trace'];resources[file.name]={'seconds':a['seconds'],'peak_RSS_bytes':max(i['peak_RSS'] for i in z),'max_MPS_driver_bytes':max(i['MPS_driver'] for i in z),'max_conservative_RSS_plus_MPS_bytes':max(i['conservative_RSS_plus_MPS'] for i in z),'max_swap_growth_bytes':max(i['swap_bytes'] for i in z)-a['initial_swap_bytes'],'min_system_free_percent':min(i['system_memory_free_percent'] for i in z)}
 evidence=['plan.json','selection-audit.json','feature-code-lock.json','training-code-lock.json','inference-code-lock.json','train-feature-validation.json','training-validation.json','models/manifest.json','test-feature-validation.json','final-inference/validation.json','reproduction-check/validation.json','README.md','finalize.py']
 out={'study':178,'candidate':'021','status':'completed_unsubmitted','UTC':datetime.now(timezone.utc).isoformat(),'CSV_absolute_path':str(csvpath),'CSV_sha256':hh,'CSV_bytes':csvpath.stat().st_size,'frozen_recipe':plan['recipe'],'explicit_reuse':plan['reuse'],'selection_audit':audit,'features':feat,'training':train,'inference':v,'second_CLI_replay_CSV_byte_exact':True,'second_CLI_replay_seconds':v2['seconds'],'resource_summaries':resources,'evidence_sha256':{n:f.sha(R/n) for n in evidence},'protected_files_unchanged':plan['protected'],'no_new_install_download_payment_upload':True,'cost':0,'holdout_subgroup_scored':False,'new_selection_or_clipping':False,'Public_score':None,'remaining_blockers':[],'limitations':['Adaptive research on repeatedly used400; small gain and conditionalCI includes0.','No new Public score;018 remains submitted best.','CPU encoder spotcheck covers3 predefinedtestimages, not1000.','018 CSV/models/testfeatures intentionally reused; only L heads/features freshly generated.','Same environment reproduction only; no second separate-directory full raw500 training run.']}
 dest=P/'reports/candidate021-178-summary.json';f.write(dest,out)
 backup=P/'reports/README-before-178.md';assert not backup.exists();backup.write_bytes((P/'README.md').read_bytes())
 with (P/'README.md').open('a') as fp:
  fp.write('\n\n## 178 candidate021 생성·검증 완료 (미제출)\n\n')
  fp.write('177에서 고정한 `0.90×018 + 0.10×L MLP3seed평균`을 그대로 적용했습니다. L은 원본500 특징과3개 최종head를 새로 학습했고, 각seed 전체학습 재실행의 텐서·예측이 정확히 일치합니다. 018 모델·CSV·176 test특징은 고정90% 성분으로 명시적으로 재사용했습니다.\n\n')
  fp.write(f'새 CSV: `artifacts/submission-candidate-021.csv`, SHA256 `{hh}`. test1000 L특징은 원본에서 새로 추출했습니다. ID/열/행수/순서/finite/CSV재로딩·CPU/MPS·018해시 보존 검증 통과, 별도 저장모델 CLI 실행도 CSV 바이트 정확히 재현했습니다. 예측범위 {v["prediction_range"]["min"]:.6f}–{v["prediction_range"]["max"]:.6f}, 클리핑 없음.\n\n')
  fp.write('비율·층·seed는177 결과 전에 고정됐으나, 같은400을 반복 사용한 두readout 선택과 재분할 확인이므로 독립 검증이 아닙니다. 개선은 작고 조건부구간에0을 포함합니다. CPU/MPS 회귀예측 비교는1000장 전체, L CPU인코더 비교는사전지정3장입니다. 기존018과보호15파일 보존,018 Public최고상태 유지. 새설치/다운로드/비용/업로드/holdout추가평가 없음. 상세 `reports/candidate021-178-summary.json`, 재현방법 `experiments/candidate021-178/README.md`.\n')
 print(json.dumps({'summary':str(dest),'summary_SHA256':f.sha(dest),'CSV':str(csvpath),'SHA256':hh,'passed':True},indent=2))
if __name__=='__main__':main()
