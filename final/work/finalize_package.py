"""Finalize measurements and local deliverables after both fixed runs finish."""
from pathlib import Path
import csv,hashlib,json,shutil,subprocess,sys,math
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=HERE.parents[2];PACK=BASE/'package'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

source=[]
for seed in [1811,1823]:
    tag=f'full500-sourceflow38-s{seed}-test';path=BASE/f'verification/source-raw/{tag}-verification.json'
    d=json.loads(path.read_text());assert d['all_rows'] and d['rows']==1000 and d['exact_array']
    data=path.with_name(tag+'.npz');ref=PACK/f'project/experiments/supervised-renewal-20261018/candidate-source-flow38-v1/features/{tag}.npz'
    d.update(feature_sha256=sha(data),reference_feature_sha256=sha(ref),feature_file_byte_identical=sha(data)==sha(ref))
    dest=PACK/'verification/source-raw';dest.mkdir(parents=True,exist_ok=True)
    (dest/path.name).write_text(json.dumps(d,indent=2)+'\n');source.append(d)
tab=json.loads((BASE/'verification/selected-tab72/verification.json').read_text());assert tab['complete'] and tab['members']==72
out=BASE/'verification/full-head-replay'
if not (out/'verification.json').exists():
    subprocess.run([sys.executable,str(PACK/'assemble_replayed.py'),'--standard',str(BASE/'verification/standard-heads-v2'),
        '--tab',str(BASE/'verification/selected-tab72'),'--source-tab',str(BASE/'verification/source-tab'),'--cnn',str(BASE/'verification/raw-cnn'),
        '--flow-dir',str(BASE/'verification/source-raw'),'--out',str(out)],cwd=ROOT,check=True)
combined=json.loads((out/'verification.json').read_text());assert combined['complete'] and combined['source_flow_regenerated']
dest=PACK/'verification/full-head-replay';dest.mkdir(parents=True,exist_ok=True)
(dest/'verification.json').write_text(json.dumps(combined,indent=2)+'\n')
primary=BASE/'submission/candidate-source-flow38-boostedreadout.csv'
expected='4e97c57b57eb77b0056552e8649f41bba5a927993b705c4602ffa272c1f3e597'
assert sha(primary)==expected
with primary.open() as f:
    reader=csv.DictReader(f);assert reader.fieldnames==['ID','hardness'];rows=list(reader)
ids=[r['ID'] for r in rows];sample=[r['ID'] for r in csv.DictReader((ROOT/'data/sample_submission.csv').open())]
assert ids==sample and len(ids)==len(set(ids))==1000
assert all(math.isfinite(float(row['hardness'])) for row in rows)
summary={'selected_submission':62,'primary_csv_sha256':expected,'public_rmse':8.8196272229,'private_rmse':9.6966884828,
    'public_private_source':'Authenticated original submission record, not a score from newly replayed predictions.',
    'primary_cached_replay_byte_exact':True,'three_hgb_refit_byte_exact':True,'raw_source_inferences':2000,
    'primary_csv_schema_ids_rows_finite_verified':True,
    'raw_source_feature_max_difference':max(d['max_abs_feature_difference'] for d in source),
    'tab_contexts_recomputed':81,'combined_prediction_max_difference':combined['max_abs_prediction_difference'],
    'combined_prediction_rms_difference':combined['rms_prediction_difference'],
    'full_auxiliary_retraining_executed':False,'raw_physical_pipeline_executed_this_task':False,
    'historical_physical_reproduction_evidence_reused':True,'protected_boundary_files_unchanged':52,
    'selected_final_files_unchanged':6,'compliance_approved':False,
    'unresolved':['Exact realdeg2 boundary replacement scope','Official large-weight delivery method/attachment limit'],
    'external_uploads_posts_messages_or_final_selection_changes':False}
(PACK/'verification-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
readme=PACK/'README.md';s=readme.read_text()
addition=('\n## 최종 독립 재현 측정\n\n'
    f'- Source encoder2개 x 원본 test1000장: 총2000회 추론,38개 특징 차이0.\n'
    f'- 상속된 TabPFN72개 + source/control9개 context 재학습 완료.\n'
    f'- 독립 readout 재현 후 전체 결합: 최대 예측 차이{combined["max_abs_prediction_difference"]:.12g}, RMS 차이{combined["rms_prediction_difference"]:.12g}.\n'
    '- P104 물리 특징과 residual CNN context는 고정 캐시다. 전체 auxiliary 모델의 처음부터 재학습은 수행하지 않았다.\n'
    '- 원래 Set raw mean의 float32 계산을 보존했다. 원본 CSV의 byte-exact 경로와 독립 모델 재계산 경로를 구분한다.\n')
if '## 최종 독립 재현 측정' not in s:readme.write_text(s+addition)
repro=PACK/'REPRODUCTION.md';s=repro.read_text()
command='''\n## 5. 독립적으로 재실행한 모델의 전체 결합\n\n```sh\npython assemble_replayed.py --standard ../runs/standard --tab ../runs/tab72 \\\n  --source-tab ../runs/source-tab --cnn ../runs/cnn --flow-dir ../runs/flow \\\n  --out ../runs/all-heads\n```\n\n이 경로의 출력은 `replayed-all-heads.csv`이며 원래 선택 파일을 대체하지 않는다. 학습/실행 backend의 부동소수점 차이를 보고한다. Public/Private 점수는 원래 선택된 CSV의 기록이다.\n'''
if '## 5. 독립적으로 재실행한 모델의 전체 결합' not in s:repro.write_text(s+command)
subprocess.run([sys.executable,str(HERE/'build_delivery.py'),'code'],cwd=ROOT,check=True)
print('FINAL MEASUREMENTS READY',json.dumps(summary),flush=True)
