"""Human-readable verified outcomes and delivery manifest; no model selection fit."""
import json
from pathlib import Path


def main():
    import cnn_experiment as e
    h=Path(__file__).resolve().parent;r=e.r
    read=lambda p:json.loads((h/p).read_text())
    final=read('final500/independent-verification.json');assert final['all_passed']
    inference=read('final500/inference-verification.json');assert inference['complete'] and not inference['uploaded']
    assert read('final500/pipeline-complete.json')['complete']
    n=read('noise-calibration/summary.json');aux=n['aux'];base=aux['addition']['baseline']['pooled']
    b61=read('baseline061-matched.json');bands=read('simultaneous-bands.json');recipe=read('final500/recipe.json')
    global_aux=read('aux-mean2/summary.json');global_cnn=read('cnn-mean2/summary.json')
    decision=dict(status='validated local experimental candidates; official references preserved',
        primary_review_candidate='073 plus noise-calibrated auxiliary residual CNN mean2',
        lowest_observed_local_candidate='replace normal-row 021 contribution with the same residual CNN branch',
        primary_reason='Retains021 and changes only noisy, non-fallback predictions; four original folds improved and one was unchanged.',
        replacement_caveat='Lower pooled local RMSE, but one fold worsens and most test rows change.',
        primary_oof_rmse=aux['addition']['candidate']['pooled'],replacement_oof_rmse=aux['replacement']['candidate']['pooled'],
        scored_scope=dict(unique_images=400,outer_folds=5,partition='original',source_restricted_augmentation=True),
        final_refit=dict(n_images=500,seeds=recipe['seeds'],production_coefficients=recipe['production_weights']),
        official_models_replaced=False,uploaded=False,
        limitations=['Noise-regime hypothesis was added after outer residual analysis.',
                     'Same400 development images have been reused by earlier research.',
                     'Current-round simultaneous intervals narrowly include zero; no independent new partition was scored.',
                     'Full500 coefficients are a fixed mean of internal calibration vectors, not a new globally scored OOF fit.'],
        next_highest_value='Fully nested confirmation on another frozen partition, rather than further tuning on these400 outcomes.',
        outputs=inference['outputs'])
    r.save_json(h/'decision.json',decision)
    rows=[('061 (동일한 수정 검증)',b61['same_original5fold_strict061']),('073 전체 기준',base),
          ('021 기여 제거 대조군',aux['replacement_vs_deletion']['baseline']['pooled']),
          ('잔차 CNN 두-seed 평균 추가',global_cnn['addition']['candidate']['pooled']),
          ('보조 학습 CNN 두-seed 평균 추가',global_aux['addition']['candidate']['pooled']),
          ('통계량 대조군 + 동일한 잡음별 보정',n['stats']['addition']['candidate']['pooled']),
          ('영상 없는 대조군 + 동일한 잡음별 보정',n['context']['addition']['candidate']['pooled']),
          ('보조 CNN + 잡음별 보정: 073에 추가',aux['addition']['candidate']['pooled']),
          ('보조 CNN + 잡음별 보정: 021 대체',aux['replacement']['candidate']['pooled'])]
    text=['# 전체073과 잔차 CNN: 실행·검증 결과','',
          f'기존073 전체 구성을 유지한 추가 후보의 OOF RMSE는 **{rows[-2][1]:.5f}**, 정상 샘플에서021 기여를 대체한 후보는 **{rows[-1][1]:.5f}**였다. 기준073은 **{base:.5f}**이다. 전체500개로 새 CNN 두 개를 재학습했고, 각1000행의 별도 제출 후보 파일을 생성·검증했다. 업로드하거나061/073을 교체하지 않았다.','',
          '검증은 기존073의 **original400개·5-fold**다. 증강본은 해당 모델의 실제 학습 원본에서만 가져오며, 실제 GMM 구성 요소를 사용했다. 이전3-partition pooled 점수9.58169 또는 축소된500개 benchmark9.75538과 이 표를 직접 비교하면 안 된다.','',
          '| 방법 | OOF RMSE | 073 대비 개선 |','|---|---:|---:|']
    for name,value in rows:text.append(f'| {name} | {value:.5f} | {base-value:+.5f} |')
    text+=['','전체073은 `0.16×G + 0.84×T`로 썼다. G는 CNN·KRR·DINO 계열 등을 포함한 완전한021 앙상블이다. T에는 TabPFN의3개 피처 구성×4개 U-Net 변형×3개 seed, Ridge, 네 변형의logRidge, 세 seed의grain-set, GMM-Ridge가 모두 들어 있다. 추가는 `073+λr`, 대체는 `T+λr`이다. 분할 실패 샘플은 모든 비교에서 기존021 예측을 유지했다.','',
          '새 CNN은 native256 입력의68,788개 파라미터 모델이다. 전체T의 정직한 교차검증 잔차를 학습하고, 보조 모델은 원래 타깃 예측 손실0.2와 여섯 개의 기존 예측·물성 피처를 추가한다. 두 seed를 고정된1:1로 평균한다. 잔차CNN과 대조용 신경망은40 epochs, AdamW, 같은 증강·체크포인트 평균·추론 규칙을 사용했다.','',
          '각 외부 학습320개 안에서 A213으로 모델을 학습하고 B107에서만 계수를 정했다. A의 잔차 타깃도 더 안쪽 교차검증으로 만들었다. 최종CNN은 외부 학습320개로 다시 학습했으며 외부80개의 정답은 평가에만 썼다. 가중치 선택에 완전한021의 A-only 체크포인트를 사용했고 기존 예측 재현 오차는0이었다.','',
          '확인된 병목은 고잡음 오차와 보정 적용 구간이었다. 보조 CNN에 전역 계수 하나를 적용하면, T 대비 고잡음 RMSE는0.44894 개선됐지만 저잡음은0.09472 악화했다. 기존073의 경계12를 유지하고, 저·고잡음 계수와 수축 강도를 B 내부 leave-one-out으로 선택했다. 저잡음 계수는 다섯 fold 모두0으로 선택됐으며 외부 결과를 보고 임의로0을 넣은 것이 아니다.','',
          '영상 없는 모델과 평균·표준편차만 쓰는 모델에도 같은 두 seed 평균과 같은 계수 선택법을 적용했다. 보조 CNN은 영상 없는 대조군보다0.18169 낮은 RMSE를 냈다. 반면 동결 ImageNet/MicroNet 표현의 잔차 Ridge는 뚜렷한 추가 이득을 만들지 못했다. 더 크거나 유명한 인코더를 추가하는 것만으로 해결되지는 않았다.','',
          '| Fold | 073 기준 | CNN 추가 | 021 대체 |','|---|---:|---:|---:|']
    baseline_f=aux['addition']['baseline']['folds'][0];add_f=aux['addition']['candidate']['folds'][0];replace_f=aux['replacement']['candidate']['folds'][0]
    for k in range(5):text.append(f'| {k} | {baseline_f[k]:.5f} | {add_f[k]:.5f} | {replace_f[k]:.5f} |')
    add=aux['addition'];replace=aux['replacement'];increment=aux['replacement_vs_deletion']
    text+=['',f'추가는4개 fold에서 개선됐고 하나는 그대로였다. 대체는4개에서 개선됐고 fold1에서는 악화했다. 대체의 총 개선{replace["gain"]:.5f} 중 **{increment["gain"]:.5f}**는021 제거 대조군에 대한 CNN의 추가 이득이다.',
          f'추가 후보의 이득 상위5개/10개 ID를 제외해도 개선은 {add["gain_without_top5_improvements"]:.5f} / {add["gain_without_top10_improvements"]:.5f} 남았다. RMSE에 기여가 큰 고잡음 사례를 줄인 결과이며, 원본 이미지나 정답·극단값은 제거하지 않았다.','',
          f'고정된 예측에 대한 paired ID bootstrap95% 개선 구간은 추가 **[{add["conditional_id_bootstrap_gain_ci95"][0]:.5f}, {add["conditional_id_bootstrap_gain_ci95"][1]:.5f}]**, 대체 **[{replace["conditional_id_bootstrap_gain_ci95"][0]:.5f}, {replace["conditional_id_bootstrap_gain_ci95"][1]:.5f}]**다. 그러나 이번{bands["comparison_count"]}개 비교의 동시 구간에서 추가는 **[{bands["bands"]["aux-noise-addition"]["simultaneous_ci95"][0]:.5f}, {bands["bands"]["aux-noise-addition"]["simultaneous_ci95"][1]:.5f}]**로0을 포함한다. 과거 연구 선택 편향과 재학습 변동까지 보정한 구간도 아니다.',
          '따라서 로컬 개선은 관측·재현됐지만 Private LB 일반화가 확정됐다고 해석하지 않는다. 잡음별 보정은 외부 잔차 구조를 본 뒤 추가한 탐색이다. 새로운 분할에서 전체 중첩 검증을 반복하는 것이 다음 우선순위다. CNN 개별 seed의 전역 보정 개선도 같은 방향이었지만, 이를 독립 검증 데이터로 세지 않았다.','',
          '검토 우선 후보는073에 CNN을 추가하는 방식이다. 원래021을 유지하고 저잡음 예측을 바꾸지 않으며, 활동한 모든fold에서 개선됐기 때문이다. 최저 로컬 점수 후보인021 대체도 별도 파일로 제공한다. 공식 주력 모델·공유 설정·기존 제출물은 변경하지 않았다.','',
          f'전체500 재학습에는 새3-fold teacher(seed2026101607)와 CNN seeds{recipe["seeds"]}를 사용했다. 생산 계수는 다섯 내부 검증 계수의 산술평균으로 사전에 고정했다. 저잡음 계수는0, 고잡음 계수는 추가{recipe["production_weights"]["addition"][1]:.9f}, 대체{recipe["production_weights"]["replacement"][1]:.9f}다. 전역 외부OOF나 테스트로 계수를 다시 맞추지 않았다. 이500개 모델 및 평균 계수에 별도의 새OOF 점수를 부여하지 않는다.','',
          '| 제출 후보 | 행 수 | 073 대비 변경 행 | 상태 |','|---|---:|---:|---|']
    for arm,label in [('addition','[073 유지 + CNN 추가](final500/candidate-aux-noise-addition.csv)'),('replacement','[021 기여 대체](final500/candidate-aux-noise-replacement.csv)')]:
        out=inference['outputs'][arm];text.append(f'| {label} | {out["n_rows"]} | {out["changed_rows"]} | 생성·검증 완료, 업로드 안 함 |')
    text+=['',f'테스트에서는 고잡음·정상 분할{inference["n_cnn_evaluated"]}개에 CNN을 적용했고, 실패{inference["fallback_count"]}개는 기존 예측을 유지했다. 모델 학습에서는 테스트 입력 읽기를 차단했고 추론에서는 학습 정답·학습 픽셀을 읽지 않았다. CSV 수식·순서·유한성·체크포인트 재로딩을 독립 검증했으며 테스트 신경망 재추론 최대 차이는 '+f'{max(v["max_abs_error"] for v in final["neural_replay"]):.3g}'+ '였다.061/073의 CSV와 manifest 해시는 모두 일치했다.','',
          '재현은 [RUN.md](RUN.md)를 따른다. [full_branch.py](full_branch.py)가 전체T 학습, [cnn_experiment.py](cnn_experiment.py)가 중첩CNN 실험, [noise_calibration.py](noise_calibration.py)가 내부 계수 선택을 담당한다. [final_teachers.py](final_teachers.py)와 [final_model.py](final_model.py)가500개 재학습·추론을 수행한다. 최종 설정·계수·학습 IDs는 [recipe.json](final500/recipe.json), 출력 검증은 [independent-verification.json](final500/independent-verification.json)에 저장했다. 기존061/073은 [보존된 재현 코드](../reproduce_best.py)로 그대로 생성할 수 있다.','',
          '전체 비교·실패 결과는 [REPORT.md](REPORT.md), [comparison.csv](comparison.csv), [fold-metrics.csv](fold-metrics.csv), [results.json](results.json)에 있다. 계획과 변경 순서는 PLAN.md와 AMENDMENT-01~06.md에 기록했다.']
    (h/'RESULT.md').write_text('\n'.join(text)+'\n')
    files=list(h.glob('*.py'))+list(h.glob('*.md'))
    files += [h/'decision.json',h/'results.json',h/'comparison.csv',h/'fold-metrics.csv',h/'simultaneous-bands.json',h/'verification.json',h/'noise-calibration/summary.json']
    files += list((h/'final500').glob('*.json'))+list((h/'final500').glob('*.csv'))+list((h/'final500/models').glob('*/model.pt'))
    r.save_json(h/'deliverable-hashes.json',{str(p.relative_to(r.ROOT)):r.digest(p) for p in files})
    print('Verified result report and delivery manifest written',flush=True)


if __name__=='__main__':
    main()
