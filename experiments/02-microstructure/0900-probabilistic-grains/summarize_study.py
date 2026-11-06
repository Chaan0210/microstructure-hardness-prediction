"""Verify and summarize completed isolated experiments, without fitting models."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
from pathlib import Path
import numpy as np
from common import HERE,ROOT,load_base,evaluate,dump_json,sha


def align(d, ref, key):
    ids=d['ids'].astype(str);pos={i:j for j,i in enumerate(ids)}
    assert set(ids)==set(ref) and len(ids)==len(set(ids))
    a=np.asarray(d[key]);return a[..., [pos[i] for i in ref]]


def run():
    b=load_base();rows=[];sources={}
    def add(name, pred, method, description):
        r=evaluate(pred,b)
        rows.append(dict(name=name,method=method,description=description,**r))
    add('045 reference',b['baseline'],'reference','Frozen existing prediction')
    rp=HERE/'ridge/predictions.npz';sources[str(rp.relative_to(HERE))]=sha(rp)
    with np.load(_repo_io(rp)) as d:
        for key,name in [('weight_only_084','021 weight-only control'),('ridge_20','Ridge 20%'),('ridge_same021share','Ridge fixed021 share'),('fgls_ridge_20','FGLS 20%')]:
            add(name,align(d,b['ids'],key),'stage1',key)
    sp=HERE/'set_model/replicated-oof.npz';sources[str(sp.relative_to(HERE))]=sha(sp)
    with np.load(_repo_io(sp)) as d:
        for key,name in [('raw_summary_ridge_blend20','Raw grain summary Ridge all3seed mean 20%'),('grain_set_blend20','Grain set all3seed mean 20%'),('grain_set_same021_share','Grain set all3seed fixed021 share')]:
            add(name,align(d,b['ids'],key),'stage3',key)
    matched=HERE/'matched-controls/predictions.npz'
    if matched.exists():
        sources[str(matched.relative_to(HERE))]=sha(matched)
        with np.load(_repo_io(matched)) as d:
            p=align(d,b['ids'],'prediction');p=np.where(b['failed'],b['p021'],p)
            add('Bare X98 with matched stage2 selection 20%',.8*b['baseline']+.2*p,'attribution_control','Same stage2 inner folds and alpha grid, no added measurements')
    stat=HERE/'probabilistic/contrast-adaptive/stage2/predictions.npz'
    if not stat.exists():raise FileNotFoundError('Stage2 is not complete')
    sources[str(stat.relative_to(HERE))]=sha(stat)
    with np.load(_repo_io(stat)) as d:
        for key in d.files:
            if '__' not in key:continue
            p=align(d,b['ids'],key)
            p=np.where(b['failed'],b['p021'],p)
            add(key+' 20%',.8*b['baseline']+.2*p,'stage2',key)
    kernel=HERE/'probabilistic/contrast-adaptive/stage4/predictions.npz'
    if not kernel.exists():raise FileNotFoundError('Stage4 is not complete')
    sources[str(kernel.relative_to(HERE))]=sha(kernel)
    with np.load(_repo_io(kernel)) as d:
        for key in d.files:
            if key in ('ids','y','baseline','fold'):continue
            if d[key].shape!=(3,400):continue
            p=align(d,b['ids'],key);p=np.where(b['failed'],b['p021'],p)
            add(key+' 20%',.8*b['baseline']+.2*p,'stage4',key)
    pp=HERE/'pfn-readout-v2/predictions.npz'
    if not pp.exists():raise FileNotFoundError('MPS PFN comparison is not complete')
    sources[str(pp.relative_to(HERE))]=sha(pp)
    with np.load(_repo_io(pp)) as d:
        for key in ('base98','phase_mean','phase_mean_and_sd'):
            p=align(d,b['ids'],key);p=np.where(b['failed'],b['p021'],p)
            add('PFN '+key+' 20%',.8*b['baseline']+.2*p,'stage2_strong_readout',key)
    dump_json(HERE/'results.json',{'rows':rows,'sources':sources,'comparison':'Same400IDs,3reusedpartitions, fixedblends. No fresh independentvalidation or newPublic/private scores.'})
    original=json.loads((HERE/'preservation-before.json').read_text())['protected_sha256']
    changes=[]
    for rel,expected in original.items():
        p=_repo_path(ROOT, rel)
        actual=sha(p) if p.is_file() else None
        if actual!=expected:changes.append({'path':rel,'expected':expected,'actual':actual})
    preservation={'checked_files':len(original),'changed_or_missing':changes,'passed':not changes}
    dump_json(HERE/'preservation-after.json',preservation)
    assert not changes,'Legacy file preservation failed'
    lines=['# Study900: original-train probabilistic measurements and grain-set learning','',
           '**판정: 큰 폭의 개선은 확인되지 않았다. 가장 나은 고정 후보는 Ridge 20% 결합으로 dev 9.940240 → 9.831835다. 8점대 달성이나 Public/private 개선은 아직 확인하지 않았다.**','',
           '기존 코드와 제출물을 보존하고 별도 경로에서 실행했다. 추가 합성 데이터와 test 학습은 없다. 기존 400개 개발 ID의 3개 분할을 재사용한 조건부 검증이며 Public/private 점수는 측정하지 않았다.','',
           '| 방법 | dev RMSE | 045 대비 개선 | 고잡음·조대/중간 RMSE |','|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['name']} | {r['rmse']:.6f} | {r['rmse_gain']:+.6f} | {r['groups']['noisy_coarse_medium']['rmse']:.6f} |")
    lines+=['','## 결과 해석','',
            '- Ridge 보완의 일부 이득은 기존021 비중 감소에서 온다. 021 비중을 유지하면 개선은 약0.063이며, 단순 비중 조정 대조군에 대한 우위는 불확실하다.',
            '- 집합 모델 최초seed 결합9.830188만 선택하지 않았다. 추가2seed를 모두 평균하면9.875014로 효과가 줄고, 고잡음 구간의 개선은 뚜렷하지 않았다. 따라서 전체500 재학습/추론으로 승격하지 않았다.',
            '- 동일한 stage2 내부 CV를 적용한 bare-X98 대조군은9.856435, GMM 추가는9.838171, 계층 확률 추가는9.928435다. 회귀기 선택 절차 차이를 분리해도 계층 모형은 도움이 되지 않았다.',
            '- MPS TabPFN은 기존98열만 쓴 비교가9.866652, 계층 분율 추가9.889573, 분율+불확실성9.879261이다. 강한 판독기에서도 새 확률 특징의 이득이 나타나지 않았다.',
            '- 확률 입력 커널은 점 입력 커널과 같은 특징/내부 CV로 비교했고 개선되지 않았다. 이는 이 측정 모형과 커널 구현의 결과이며, 모든 확률적 방법이나 원본 영상의 정보 한계를 증명하지 않는다.',
            '- 제출용 CSV로 생성하고 재현 검증한 후보는 Ridge 20%다. 기존045를 자동 교체하거나 온라인 제출하지 않았다.']
    lines+=['','## 검증과 해석','',
            '- 원본 500장과 고정 제로샷 마스크에서 결정립 관측을 추출했다. 개별 결정립은 독립 경도 표본으로 복제하지 않았다.',
            '- 신규 측정 사전분포·스케일러·회귀기·신경망 조기종료는 outer training 안에서만 학습/선택했다.',
            '- 기존 X98 및 045 자체는 과거에 개발한 표현이므로 완전히 새로운 검증이라고 부를 수 없다.',
            '- 최초 확률 모델의 단일 최적점 적분은 다중 해를 놓쳐 무효 처리했다. 해당 파일은 삭제하지 않고 INVALID 표기로 보존했다.',
            '- 최종 다중 최적점 적분은 독립 적분 기준과 비교했다. 공유 음영·분포 모형·공분산 단순화는 여전히 가정이다.',
            '- 집합 모델은 최초 seed만 선택하지 않고 사전 고정한 추가 2개를 포함해 모두 평균했다.',
            '- 작은 집합망은 벤치마크에서 CPU가 더 빨랐다. 대형 TabPFN 판독기는 MPS에서 실행했다.',
            '- 최종 확률 특징:60캐시/20,400추론 전부 수렴, 학습 범위 검증. 드문6사례만 경계에 안전한 Beta-Jacobi 및 적응 적분을 사용했다.',
            '- MPS의 phase+SD 저장 context는 같은 MPS에서80개 예측이 정확히 재현됐다. CPU 내 반복도 정확하나 CPU-MPS 사이 RMS0.028465HV 차이는 남는다(20%결합 RMS0.005693HV).',
            f"- 기존 코드·문서·제출물 {len(original):,}개 SHA256 재검사: 변경/삭제 0개.",
            '', '## 실행 파일','',
            '- `ridge/run.py`: Ridge 및 FGLS 대조.',
            '- `extract_objects.py`: 원본 화소 결정립 데이터.',
            '- `probabilistic/contrast-adaptive/run_stage2.py`: 수정된 확률 측정.',
            '- `set_model/run.py`, `set_model/run_replication.py`: 집합 모델 및 3seed 확인.',
            '- `probabilistic/contrast-adaptive/run_stage4.py`: 점 입력/확률 입력 커널 대조.',
            '- `pfn_readout.py`: MPS TabPFN 동일 조건 비교.',
            '- `verify_mps_context.py`: 같은 MPS에서 저장 context 재현.',
            '- `matched_ridge_control.py`: 동일 내부 CV의 bare-X98 귀속 대조.',
            '- `ridge/finalize500.py`: 검증한 고정 Ridge 결합의 전체500 학습 및 로컬 추론.',
            '', '전체 수치, 구간별 오차, 분할별 결과와 조건부 ID bootstrap은 `results.json`에 저장했다.',
            '로컬 CSV는 `ridge/final500/candidate-045-ridge20.csv`에 있으며 제출하지 않았다.',
            '', 'MPS 비교를 새 출력 경로에 재현하는 명령(저장소 루트에서 실행):', '',
            '```sh',
            'experiments/tabpfn-v2/.venv/bin/python -B experiments/probabilistic-grains-900/pfn_readout.py --features experiments/probabilistic-grains-900/probabilistic/contrast-adaptive/stage2/features --output experiments/probabilistic-grains-900/pfn-readout-reproduction --device mps',
            '```', '',
            '현재 진입점은 무효 처리한 원래 캐시와 완료된 결과 덮어쓰기를 거부한다. 실제 실험 당시 소스는 `pfn_readout_executed.py`에 SHA 그대로 보존했고, 기본 경로/입출력 보호 변경은 `pfn-readout-v2/entrypoint-repair.json`에 기록했다. 모델 생성과 특징 조립의 AST는 동일하다.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':True,'rows':len(rows),'preservation':preservation,'results':str(HERE/'RESULTS.md')},ensure_ascii=False))


if __name__=='__main__':run()
