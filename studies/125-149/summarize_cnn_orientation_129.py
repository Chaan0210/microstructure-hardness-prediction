"""Record the predeclared screen, conditional paired uncertainty and preservation."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_orientation_129 as s
import numpy as np
e=s.e

def main():
    plan=s.check();decision=e.read(s.OUT/'screen-decision.json');splits=list(e.SPLITS) if decision['advance'] else ['original'];records={};validation={};data=[]
    for split in splits:
        r=e.read(s.OUT/split/'report.json');v=e.read(s.OUT/f'validation-{split}.json');assert v['passed'] and v['report_sha256']==e.sha(s.OUT/split/'report.json');validation[split]=v
        with np.load(_repo_io(s.OUT/split/'predictions.npz')) as z:
            if data:np.testing.assert_array_equal(z['ids'],data[0]['ids']);np.testing.assert_array_equal(z['y'],data[0]['y'])
            d={key:z[key].copy() for key in ['ids','y','fold','cnn','raw']};data.append(d)
        records[split]={'scores':r['scores'],'main_fit_seconds':r['seconds'],'fold_RMSE_gains':{key:[e.rmse(d['y'][d['fold']==k],d[key][0,d['fold']==k])-e.rmse(d['y'][d['fold']==k],d[key][1,d['fold']==k]) for k in range(5)] for key in ['cnn','raw']},'report_sha256':e.sha(s.OUT/split/'report.json')}
    rng=np.random.default_rng(20271111);ix=rng.integers(0,400,(10000,400));contrasts={};boot={};pooled={};train={}
    for key in ['cnn','raw']:
        loss=np.stack([(d[key]-d['y'])**2 for d in data]).mean(axis=0);metric=np.sqrt(loss.mean(axis=1));g=np.sqrt(loss[0,ix].mean(1))-np.sqrt(loss[1,ix].mean(1));independent=[]
        for row in ix:
            counts=np.bincount(row,minlength=400);independent.append(math.sqrt(float(counts@loss[0])/400)-math.sqrt(float(counts@loss[1])/400))
        error=float(np.max(abs(g-np.array(independent))));assert error<1e-12
        contrasts[key]={'RMSE_gain_D4_minus_identity':float(metric[0]-metric[1]),'MSE_gain':float(loss[0].mean()-loss[1].mean()),'conditional_gain_95':np.percentile(g,[2.5,97.5]).tolist(),'bootstrap_independent_max_error':error};boot[key]=g;pooled[key]=dict(zip(s.ARMS,metric.tolist()))
    for arm in s.ARMS:
        errors=[]
        for split in splits:
            r=e.read(s.OUT/split/'report.json');errors.extend(x['arms'][arm]['train_RMSE']**2 for x in r['folds'])
        train[arm]=float(np.sqrt(np.mean(errors)))
    with (s.OUT/'conditional-bootstrap.npz').open('xb') as f:np.savez_compressed(_repo_io(f),**boot)
    consistent=all(records[split]['scores']['identity'][m]<records[split]['scores']['D4'][m] for split in splits for m in ['CNN_RMSE','raw4way_RMSE']) and len(splits)==3
    recommendation=('Both endpoints improved on all3 reused partitions. Next eligible step is new nested identity CNN fits and downstream affine/physics-residual refits. No direct swap into014 or deployment authorized by these raw scores alone.' if consistent else 'Retain014 and close this exact identity-only CNN hypothesis. No resplit/augmentation-ratio/subgroup/seed rescue search and no downstream/full500 retraining justified by this screen.' if not decision['advance'] else 'Retain014: original improvement did not transfer consistently to both reused resplits. Do not integrate or search alternative augmentation settings.')
    old=e.read(_repo_path(ROOT, 'artifacts/problem-definition-111/plan.json'))['existing_report_sha256']
    for f,h in old.items():assert e.sha(_repo_path(ROOT, f))==h,f
    packages=len(s.q.a.check()['environment']);s.check()
    result={'study':'129 CNN identity versus D4 training','status':'completed','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rationale':plan['hypothesis'],'novelty':plan['novelty'],'partitions_completed':splits,'screen':decision,'results':records,'pooled_validation_RMSE':pooled,'pooled_training_RMSE':train,'contrasts':contrasts,'all3_consistent':consistent,'recommendation':recommendation,'interpretation_limits':['Removing D4 changes both orientation information and augmentation regularization. Accuracy cannot alone establish the physical/generator invariance or dependence.','No fabricated labels, interpolated images/labels, or newly assumed label-preserving transforms in identity arm. D4 retained only as existing matched control.','Raw4way excludes full014 affine/shortB/residual; no full014 improvement or degradation claimed.','Only400 historically reused images and one paired seed per fold. Bootstrap is conditional fixed-fit uncertainty and not selection-adjusted/independent confirmation.'],'validation':validation,'plan_sha256':e.sha(s.OUT/'plan.json'),'bootstrap_sha256':e.sha(s.OUT/'conditional-bootstrap.npz'),'execution':{'main_fits':10*len(splits),'all_fits_including_label_intervention':10*len(splits)+1,'main_seconds':sum(r['main_fit_seconds'] for r in records.values()),'device':'MPS','CPU_threads':2,'new_installs':0,'new_data_or_weights':0,'cost':0,'holdout_decoded_or_scored':False,'test_decoded_inferred_or_fit':False,'existing_protected_artifacts_hashed_only':True,'evaluation_server_zeroing':False,'new_submission_CSV':False,'uploaded':False},'preservation':{'014_CSV_SHA256':e.sha(s.q.p.CSV),'previous_reports_verified':len(old),'packages_unchanged':packages},'execution_blockers':[],'scientific_unknowns':['Exact hardness proxy formula and its absolute-orientation dependence remain unknown.'],'source_sha256':e.sha(Path(__file__))}
    dest=_repo_path(ROOT, 'reports/cnn-orientation-129-summary.json');e.write(dest,result)
    readme=_repo_path(ROOT, 'README.md');before=e.sha(readme);assert before==e.read(_repo_path(ROOT, 'reports/readme-update-128.json'))['after_sha256'];backup=_repo_path(ROOT, 'reports/README-before-129.md')
    with backup.open('xb') as f:f.write(readme.read_bytes())
    text='\n\n## CNN 회전·반사 증강 가정 제거129 (완료, 014 보존)\n\n111에서 확인한 합성 hardness의 D4 불변 보증 부재를 실제 학습 대조로 검사했습니다.\n046의 DINO 회전 평균,110의 잡음 추가,128의 학습 예산 변경과 달리 CNN의 기존 D4를 제거하는 한 실험입니다.\n동일 TinyCNN128·80epoch·800update·초기시드·배치32·epoch cosine·마지막5epoch 평균을 사용했습니다.\nidentity도 D4 난수를 소비한 뒤 변환만 생략하여 모든 배치 순서와 dropout 난수 진행을 맞췄습니다.\n학습320개에서만 픽셀·타깃 정규화했고 양쪽 추론은 원래 방향만 사용했습니다.\n\n| 분할 | D4 CNN | identity CNN | D4 고정4way | identity 고정4way |\n|---|---:|---:|---:|---:|\n'
    for split in splits:
        sc=records[split]['scores'];text+=f"| {split} | {sc['D4']['CNN_RMSE']:.6f} | {sc['identity']['CNN_RMSE']:.6f} | {sc['D4']['raw4way_RMSE']:.6f} | {sc['identity']['raw4way_RMSE']:.6f} |\n"
    text+=f"\n단독 CNN과 고정4way 둘 다 원본에서 개선할 때만 두 재분할로 확대하는 기준을 사전 고정했습니다.\n실행 분할: {', '.join(splits)}. 원본 통과: {decision['advance']}. 세 분할 모두 일관 개선: {consistent}.\n"
    text+='고정4way는 .3compact26+.3CNN+.2Smean3+.2Bmean3이며014의 affine·shortB·잔차 보정을 포함하지 않습니다.\n'
    for key,c in contrasts.items():text+=f"{key} pooled RMSE 개선량(D4−identity): {c['RMSE_gain_D4_minus_identity']:.6f}, 조건부95%구간 {c['conditional_gain_95']}.\n"
    text+=f"\n본 학습 {10*len(splits)}개와 학습 밖 라벨 변경 재학습1개를 실행했습니다. 기존 D4 가중치·예측 재현,\n모든 모델 MPS/CPU 재로딩, 배치·LR·5개 state 평균·실제 입력 표본 SHA·학습/검증 그룹 분리를 독립 확인했습니다.\n80개 외부 타깃 변경 재학습은 가중치 SHA가 동일했습니다.\n권고: {recommendation}\n\nD4 제거는 방향 정보와 증강 정규화 효과를 함께 바꿉니다. 결과가 물리적 회전 불변/의존성을 입증하지 않습니다.\n같은400개 반복 연구이며 bootstrap은 재학습·탐색 선택 편향을 반영하지 않습니다.\n014·이전보고서200개·패키지26개를 보존했습니다. 설치·비용·외부자료·holdout/test·서버zeroing·업로드는 없습니다.\n상세: `reports/cnn-orientation-129-summary.json`, `artifacts/cnn-orientation-129/plan.json`.\n"
    with readme.open('a') as f:f.write(text)
    e.write(_repo_path(ROOT, 'reports/readme-update-129.json'),{'before_sha256':before,'after_sha256':e.sha(readme),'backup':str(backup.relative_to(ROOT)),'reason':'Record fixed CNN D4-removal129 and matched evidence, preserve014'})
    s.check();pres={**result['preservation'],'passed':True,'all_frozen_sources_inputs_and_models_verified':True,'summary_sha256':e.sha(dest),'README_sha256':e.sha(readme),'cost':0,'uploaded':False};e.write(_repo_path(ROOT, 'reports/cnn-orientation-129-preservation.json'),pres);e.emit({k:result[k] for k in ['status','partitions_completed','screen','pooled_validation_RMSE','pooled_training_RMSE','contrasts','recommendation','execution','preservation']})
if __name__=='__main__':main()
