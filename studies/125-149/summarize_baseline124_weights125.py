"""Finalize development-only diagnostics and the single predeclared weight comparison."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, datetime
from pathlib import Path
ROOT = _RepoPath(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import constrained_weights_125 as q
import numpy as np

e = q.e

def main():
    plan = q.check()
    diag_path = _repo_path(ROOT, 'artifacts/baseline-diagnostics-124/report.json')
    diag = e.read(diag_path)
    assert e.sha(_repo_path(ROOT, 'baseline_diagnostics_124.py')) == diag['source_sha256']
    for name, h in diag['models_sha256'].items():
        assert e.sha(_repo_path(ROOT, name)) == h, name
    diag_np = diag_path.parent/'predictions.npz'
    assert e.sha(diag_np) == diag['predictions_sha256']
    P, Y, weights, anchors, validations, evidence = [], [], [], [], {}, {}
    ids0 = None
    with np.load(_repo_io(diag_np)) as dz:
        for split in e.SPLITS:
            folder = q.OUT/split
            rec = e.read(folder/'report.json')
            v = e.read(q.OUT/f'validation-{split}.json')
            assert v['passed'] and v['report_sha256'] == e.sha(folder/'report.json')
            validations[split] = v
            assert e.sha(folder/'predictions.npz') == rec['predictions_sha256']
            with np.load(_repo_io(folder/'predictions.npz')) as z:
                ids = z['ids'].tolist()
                assert ids == dz['ids'].tolist() and (ids0 is None or ids == ids0)
                ids0 = ids
                np.testing.assert_array_equal(z['y'], dz[split+'_y'])
                if Y:
                    np.testing.assert_array_equal(z['y'], Y[0])
                pred = np.stack([z['reference014'], z['primary']])
                assert pred.shape == (2,400) and np.isfinite(pred).all()
                P.append(pred); Y.append(z['y'].copy())
            for k in range(5):
                f = folder/f'fold-{k}/report.json'
                assert e.sha(f) == rec['fold_reports_sha256'][f'fold-{k}/report.json']
                fr = e.read(f)
                weights.append(fr['outer_meta'])
                anchors.extend(fr['anchor_meta'])
            for f in (folder/'report.json', folder/'predictions.npz', q.OUT/f'validation-{split}.json'):
                evidence[str(f.relative_to(ROOT))] = e.sha(f)
        bp = np.stack([dz[s+'_predictions'] for s in e.SPLITS])
        by = np.stack([dz[s+'_y'] for s in e.SPLITS])
        for j, name in enumerate(diag['error_correlation_order']):
            val = float(np.sqrt(np.mean((bp[:,j]-by)**2)))
            assert abs(val-diag['component_scores'][name]['pooled_validation_RMSE']) < 1e-12
    P, Y = np.stack(P), np.stack(Y)
    E = (P-Y[:,None,:])**2
    perid = E.mean(0)
    mse = perid.mean(1)
    R = np.sqrt(mse)
    # A resampled ID carries its three partition predictions together.
    draw = np.random.default_rng(20271103).integers(0,400,(10000,400))
    delta = np.sqrt(perid[1,draw].mean(1))-np.sqrt(perid[0,draw].mean(1))
    ci = np.quantile(delta,[.025,.975]).tolist()
    rows = []
    for i, split in enumerate(e.SPLITS):
        rr = np.sqrt(E[i].mean(1))
        dd = np.sqrt(E[i,1,draw].mean(1))-np.sqrt(E[i,0,draw].mean(1))
        rows.append({'split':split,'fixed014_RMSE':float(rr[0]),'learned125_RMSE':float(rr[1]),'RMSE_change':float(rr[1]-rr[0]),'paired_ID_conditional95_change_CI':np.quantile(dd,[.025,.975]).tolist()})
    def weight_stats(models):
        w = np.array([m['weight_model']['weights'] for m in models])
        return {'count':len(models),'order':list(q.COLS),'mean':w.mean(0).tolist(),'min':w.min(0).tolist(),'max':w.max(0).tolist(),'mean_L2_distance_from_prior':float(np.linalg.norm(w-q.PRIOR,axis=1).mean()),'mean_training_objective_improvement':float(np.mean([m['weight_model']['prior_objective']-m['weight_model']['objective'] for m in models])),'KKT_max_error':max(m['weight_model']['KKT_max_error'] for m in models),'affine_slope_range':[min(m['calibrator']['slope'] for m in models),max(m['calibrator']['slope'] for m in models)]}
    fixtures = e.read(q.OUT/'fixture-control-validation.json')
    replay = e.read(q.OUT/'replay-validation.json')
    assert fixtures['passed'] and replay['passed']
    assert mse[1] > mse[0], 'Observed result changed; reconsider the decision before publication.'
    old = e.read(_repo_path(ROOT, 'artifacts/problem-definition-111/plan.json'))['existing_report_sha256']
    for name, h in old.items():
        assert e.sha(_repo_path(ROOT, name)) == h, name
    for f in (diag_path, diag_np, q.OUT/'plan.json',q.OUT/'fixture-control-validation.json',q.OUT/'replay-validation.json'):
        evidence[str(f.relative_to(ROOT))] = e.sha(f)
    summary = {
        'study':'124 baseline diagnostics and 125 constrained weights',
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'complete; retain fixed014; reject125 on primary pooled development MSE',
        'baseline124':{'component_scores':diag['component_scores'],'actual_lineage':diag['actual_lineage'],'feature_stability':diag['feature_stability'],'training_size_diagnostic':{k:v for k,v in diag['same_outer80_training_size_diagnostic'].items() if k!='folds'},'verification':diag['verification'],'models_reloaded_bundles':len(diag['models_sha256']), 'interpretation':'ET has a large train-validation gap, while range violations are uncommon and nearly uncorrelated with per-ID squared error. This supports investigating overfitting and redundancy, but does not identify a unique causal reason for014 gains; removing baseline also changed weights, calibrators and residual targets.'},
        'weights125':{'prior':q.PRIOR.tolist(),'lambda':q.LAMBDA,'shortB_weight_fixed':.1,'pooled_MSE':{'fixed014':float(mse[0]),'learned125':float(mse[1])},'pooled_RMSE':{'fixed014':float(R[0]),'learned125':float(R[1])},'RMSE_change':float(R[1]-R[0]),'ID_cluster_conditional95_change_CI':ci,'per_partition':rows,'outer_weight_statistics':weight_stats(weights),'anchor_weight_statistics':weight_stats(anchors),'nesting':plan['nested_scope'],'duplicate_audit':plan['duplicate_audit']},
        'validation':{'partitions':validations,'independent_weight_affine_fits':sum(x['independent_weight_affine_fits'] for x in validations.values()),'residual_models_reloaded_including_none':sum(x['residual_models_reloaded'] for x in validations.values()),'fixtures_and_fixed014_refit':fixtures,'label_interventions':replay},
        'uncertainty':'10000 paired ID-cluster bootstrap draws, seed20271103. Each ID retains all three partition predictions. Models are held fixed; intervals exclude research selection and refitting variability. The three partitions reuse the same400 observations, not1200 independent examples.',
        'decision':{'keep_candidate':'014','new_submission':False,'full500_refit':False,'new_test_inference':False,'reason':'The only predeclared learned-weight procedure has worse pooled untrimmed MSE. Do not expand lambda search or blend after observing these results.','next_basis':'Wait for already prepared014 submission evidence through the parent. This result does not establish that every possible constrained weighting scheme fails. Any future representation or robustness hypothesis needs a separate fixed design and valid train-only cache lineage.'},
        'limits':plan['limits']+diag['interpretation_limits'],
        'preservation':{'candidate014_CSV_SHA256':e.sha(q.protected.CSV),'previous_reports_verified':len(old),'packages_unchanged':len(q.a.check()['environment']),'protected_sources_inputs_and014_hashes_verified':True},
        'scope':{'development_examples':400,'new_neural_fits':0,'test_or_holdout_decoded':False,'global_outer_OOF_weights_fit':False,'new_packages':0,'cost':0,'uploaded':False},
        'evidence_sha256':evidence,'source_sha256':e.sha(Path(__file__)), 'execution_blockers':[]}
    out = _repo_path(ROOT, 'reports/baseline124-weights125-summary.json')
    assert not out.exists()
    e.write(out,summary)
    readme = _repo_path(ROOT, 'README.md')
    before = e.sha(readme)
    assert before == e.read(_repo_path(ROOT, 'reports/readme-update-122-candidate014.json'))['after_sha256']
    backup = _repo_path(ROOT, 'reports/README-before-124-125.md')
    with backup.open('xb') as f:
        f.write(readme.read_bytes())
    text = '''\n\n## baseline 진단124 및 제한된 가중치 학습125 (완료, 014 유지)\n\n기존 baseline 15개 outer·45개 inner 묶음을 재로딩해 개발400개에서 진단했습니다.\nExtraTrees 학습 RMSE1.4378/검증14.7760, baseline 혼합은7.1957/14.1906입니다.\n전체795개 입력 중 ET·base Ridge는519개를 선택하고 morph Ridge는795개 모두를 사용합니다.\n검증값의 학습 min/max 이탈률은0.5414%, 개체별 이탈률과 제곱오차의 Spearman 상관은0.0206입니다.\n단순 특징 범위 이상이 주원인이라는 근거는 약합니다. 과적합·예측 중복·기존 비중을 함께 고려해야 하며014 개선의 단일 원인을 확정하지 않습니다.\n\n125는014의 compact26 KRR/CNN/Smean/Bmean 네 가중치만 비음수·합1로 학습했습니다.\n기존[0.3,0.3,0.2,0.2]에 대한 제곱 규제 lambda1 하나를 고정했고 shortB10%는 유지했습니다.\nOuter 가중치는 내부 OOF에서, 잔차 앵커 가중치는 한 단계 더 깊은 OOF에서 학습했습니다.\n전역 outer OOF는 평가에만 사용했습니다. 모든 affine과 honest 잔차를 새 혼합에 맞춰 재학습했습니다.\n074의 단일 affine,077의 무규제 macro contrast,112의 attention 혼합 비중과 구별됩니다.\n\n| 구성 | 원본 RMSE | 재분할1 | 재분할2 | pooled RMSE |\n|---|---:|---:|---:|---:|\n'''
    for label,key,j in [('고정014','fixed014_RMSE',0),('가중치 학습125','learned125_RMSE',1)]:
        vals=[x[key] for x in rows]
        text += f'| {label} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {R[j]:.6f} |\n'
    text += f'''\n125의 pooled RMSE 변화는 {R[1]-R[0]:+.6f}, 조건부 ID 묶음95% 구간은[{ci[0]:.6f}, {ci[1]:.6f}]입니다.\n같은400개를 반복 사용한 탐색이므로 이 구간은 연구 선택·재학습 불확실성을 반영하지 않습니다.\n주지표 pooled MSE가 악화돼014를 유지합니다. lambda 재탐색이나 사후 혼합을 추가하지 않습니다.\n\n가중치·affine60개를 독립 SLSQP/OLS로 검증했고 잔차 모델720개(none 포함)를 재로딩·독립 계산했습니다.\n고정014 전체 fold 재학습 대조, outer 검증 타깃 및 잔차 내부 검증 타깃 변경 검증을 통과했습니다.\n새 설치·비용·신경망 학습·holdout/test 열람·전체500 재학습·새 제출 CSV·업로드는 없습니다.\n기존014 CSV SHA256: `{e.sha(q.protected.CSV)}`\n014의 업로드 및 결과 확인은 부모 작업에서 조율합니다. 실행상 막힘은 없습니다.\n상세: `reports/baseline124-weights125-summary.json`, `artifacts/baseline-diagnostics-124/report.json`, `artifacts/constrained-weights-125/plan.json`.\n'''
    with readme.open('a') as f:
        f.write(text)
    e.write(_repo_path(ROOT, 'reports/readme-update-124-125.json'),{'before_sha256':before,'after_sha256':e.sha(readme),'backup':str(backup.relative_to(ROOT)),'reason':'Document completed124 diagnostics and failed125 fixed regularized weighting; preserve014'})
    q.check()
    preservation={**summary['preservation'],'passed':True,'summary_sha256':e.sha(out),'README_SHA256':e.sha(readme),'new_test_or_holdout_access':False,'uploaded':False}
    e.write(_repo_path(ROOT, 'reports/baseline124-weights125-preservation.json'),preservation)
    e.emit({'pooled_RMSE':summary['weights125']['pooled_RMSE'],'change':summary['weights125']['RMSE_change'],'CI':ci,'outer_weights':summary['weights125']['outer_weight_statistics'],'preservation':preservation})

if __name__ == '__main__':
    q.protected.torch.set_num_threads(2)
    main()
