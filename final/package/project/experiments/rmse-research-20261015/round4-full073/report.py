"""Summarize completed matched experiments without fitting new outer weights."""
import csv
import json
from pathlib import Path
import numpy as np
import cnn_experiment as e
HERE=e.HERE;r=e.r


def main():
    done=json.loads((HERE/'pipeline-completion.json').read_text());assert done['completed']
    folders=[p for p in sorted([*HERE.glob('*-seed[01]'),*HERE.glob('*-mean2')]) if (p/'summary.json').exists()]
    assert all((p/'verification.json').exists() for p in folders)
    data=dict(np.load(HERE.parent/'inputs.npz'));dev=data['dev'];y=data['y'][dev];fold=data['fold'][:1]
    first=np.load(HERE/'cnn-seed0/oof.npz');base=first['base'];delete=first['non021']
    predictions={'073':base,'remove021':delete};rows=[];foldrows=[];selection=[];comparisons={}
    pairs={'remove021_vs073':(base,delete)}
    for name,pred in predictions.items():
        rows.append(dict(model=name,arm='reference',rmse=r.rmse(pred,y),gain_vs073=r.rmse(base,y)-r.rmse(pred,y)))
    for folder in folders:
        summary=json.loads((folder/'summary.json').read_text());oof=np.load(folder/'oof.npz')
        assert np.array_equal(oof['base'],base) and np.array_equal(oof['ids'],data['ids'][dev])
        for arm in ('addition','replacement'):
            name=folder.name+'-'+arm;predictions[name]=oof[arm]
            q=summary[arm];control=base if arm=='addition' else delete
            additional=r.compare(control[None],oof[arm][None],y,fold)
            comparisons[name]=additional
            pairs[name]=(control,oof[arm])
            rows.append(dict(model=folder.name,arm=arm,rmse=q['candidate']['pooled'],gain_vs073=q['gain'],
                             gain_vs_matched_control=additional['gain'],improved_folds=additional['improved_folds'],
                             ci_low=additional['conditional_id_bootstrap_gain_ci95'][0],
                             ci_high=additional['conditional_id_bootstrap_gain_ci95'][1]))
        for k in range(5):
            z=np.load(folder/f'fold{k}-selection.npz');B=z['B'];valid=z['eligible'];rr=z['residual'][valid]
            rng=np.random.default_rng(20261016+k);ii=rng.integers(0,len(rr),(5000,len(rr)))
            for arm,key in [('addition','base073'),('replacement','non021')]:
                residual=(data['y'][B]-z[key])[valid];den=(rr[ii]**2).sum(1)
                ww=np.clip(np.divide((rr[ii]*residual[ii]).sum(1),den,out=np.zeros(len(ii)),where=den>1e-12),0,1)
                selection.append(dict(model=folder.name,fold=k,arm=arm,
                    weight=summary['inner_weights'][k][arm],inner_weight_bootstrap_low=float(np.quantile(ww,.025)),
                    inner_weight_bootstrap_high=float(np.quantile(ww,.975)),
                    fraction_zero=float((ww==0).mean()),fraction_one=float((ww==1).mean())))
    adaptive_path=HERE/'adaptive-weights/summary.json'
    if adaptive_path.exists():
        adaptive_reports=json.loads(adaptive_path.read_text())
        reference=np.load(HERE/'adaptive-weights/cnn-seed0.npz')['weight_only']
        predictions['adaptive021']=reference
        pairs['adaptive021_vs073']=(base,reference)
        rows.append(dict(model='adaptive021',arm='weight-only control',rmse=r.rmse(reference,y),
                         gain_vs073=r.rmse(base,y)-r.rmse(reference,y)))
        for model,report in adaptive_reports.items():
            name=model+'-joint';predictions[name]=np.load(HERE/f'adaptive-weights/{model}.npz')['joint']
            q=report['vs073'];additional=report['vs_weight_only'];comparisons[name]=additional
            pairs[name]=(reference,predictions[name])
            rows.append(dict(model=model,arm='joint inner weights',rmse=q['candidate']['pooled'],gain_vs073=q['gain'],
                             gain_vs_matched_control=additional['gain'],improved_folds=additional['improved_folds'],
                             ci_low=additional['conditional_id_bootstrap_gain_ci95'][0],
                             ci_high=additional['conditional_id_bootstrap_gain_ci95'][1]))
    frozen=HERE/'frozen-heads/summary.json'
    if frozen.exists():
        frozen_reports=json.loads(frozen.read_text())
        for family in ('imagenet','micronet'):
            assert json.loads((HERE/f'frozen-heads/{family}-verification.json').read_text())['all_passed']
            for mode in ('direct','residual'):
                model=f'{family}-{mode}';data_f=np.load(HERE/f'frozen-heads/{model}-oof.npz')
                assert np.array_equal(data_f['baseline'],base)
                for arm in ('addition','replacement'):
                    name=model+'-'+arm;predictions[name]=data_f[arm]
                    control=base if arm=='addition' else delete
                    additional=r.compare(control[None],data_f[arm][None],y,fold)
                    q=frozen_reports[model][arm];comparisons[name]=additional;pairs[name]=(control,data_f[arm])
                    rows.append(dict(model=model,arm=arm,rmse=q['candidate']['pooled'],gain_vs073=q['gain'],
                                     gain_vs_matched_control=additional['gain'],improved_folds=additional['improved_folds'],
                                     ci_low=additional['conditional_id_bootstrap_gain_ci95'][0],
                                     ci_high=additional['conditional_id_bootstrap_gain_ci95'][1]))
            for arm in ('addition','replacement'):
                pairs[f'{family}_residual_vs_direct_{arm}']=(predictions[f'{family}-direct-{arm}'],predictions[f'{family}-residual-{arm}'])
    noise_file=HERE/'noise-calibration/summary.json'
    noise_reports=json.loads(noise_file.read_text()) if noise_file.exists() else {}
    for family,noise_report in noise_reports.items():
        assert json.loads((HERE/f'noise-calibration/{family}-verification.json').read_text())['all_passed']
        n=np.load(HERE/f'noise-calibration/{family}-oof.npz');model=family+'-noise'
        assert np.array_equal(n['baseline'],base)
        for arm in ('addition','replacement'):
            name=model+'-'+arm;predictions[name]=n[arm];control=base if arm=='addition' else delete
            q=noise_report[arm];additional=r.compare(control[None],n[arm][None],y,fold)
            comparisons[name]=additional;pairs[name]=(control,n[arm])
            global_prediction=np.load(HERE/f'{family}-mean2/oof.npz')[arm]
            pairs[f'{family}_noise_vs_global_{arm}']=(global_prediction,n[arm])
            rows.append(dict(model=model,arm=arm,rmse=q['candidate']['pooled'],gain_vs073=q['gain'],
                             gain_vs_matched_control=additional['gain'],improved_folds=additional['improved_folds'],
                             gain_vs_global_calibration=noise_report[arm+'_vs_global_weight']['gain'],
                             ci_low=additional['conditional_id_bootstrap_gain_ci95'][0],
                             ci_high=additional['conditional_id_bootstrap_gain_ci95'][1]))
    for name,p in predictions.items():
        for k in range(5):
            mask=fold[0]==k
            foldrows.append(dict(model=name,fold=k,n=int(mask.sum()),rmse=r.rmse(p[mask],y[mask]),
                                 gain_vs073=r.rmse(base[mask],y[mask])-r.rmse(p[mask],y[mask])))
    cross={}
    for name in ('cnn','aux'):
        a=HERE/f'{name}-seed0/oof.npz';s=HERE/'stats-seed0/oof.npz'
        for arm in ('addition','replacement'):
            cross[f'{name}_vs_stats_{arm}']=r.compare(np.load(s)[arm][None],np.load(a)[arm][None],y,fold)
            pairs[f'{name}_vs_stats_{arm}']=(np.load(s)[arm],np.load(a)[arm])
    if (HERE/'context-seed0/oof.npz').exists():
        for arm in ('addition','replacement'):
            cross[f'aux_vs_context_{arm}']=r.compare(np.load(HERE/'context-seed0/oof.npz')[arm][None],
                np.load(HERE/'aux-seed0/oof.npz')[arm][None],y,fold)
            pairs[f'aux_vs_context_{arm}']=(np.load(HERE/'context-seed0/oof.npz')[arm],np.load(HERE/'aux-seed0/oof.npz')[arm])
    if frozen.exists():
        for family in ('imagenet','micronet'):
            for arm in ('addition','replacement'):
                old=np.load(HERE/'stats-seed0/oof.npz')[arm];new=predictions[f'{family}-residual-{arm}']
                name=f'{family}_residual_vs_stats_{arm}'
                pairs[name]=(old,new);cross[name]=r.compare(old[None],new[None],y,fold)
    if all(k in noise_reports for k in ('cnn','aux','stats','context')):
        for model,control in [('cnn','stats'),('aux','context')]:
            for arm in ('addition','replacement'):
                old=predictions[f'{control}-noise-{arm}'];new=predictions[f'{model}-noise-{arm}']
                name=f'{model}_noise_vs_{control}_noise_{arm}'
                pairs[name]=(old,new);cross[name]=r.compare(old[None],new[None],y,fold)
    # Frozen-prediction simultaneous bands for the comparisons in this round.
    # They do not repair earlier model-search bias or account for refitting noise.
    indices=np.random.default_rng(r.SEED).integers(0,len(y),(10000,len(y)))
    boot=[];point=[];pair_names=list(pairs)
    for old,new in pairs.values():
        a=(old-y)**2;b=(new-y)**2
        point.append(float(np.sqrt(a.mean())-np.sqrt(b.mean())))
        boot.append(np.sqrt(a[indices].mean(1))-np.sqrt(b[indices].mean(1)))
    boot=np.array(boot);point=np.array(point);se=boot.std(1,ddof=1);active=se>1e-12
    critical=float(np.quantile(np.max(np.abs((boot[active]-point[active,None])/se[active,None]),axis=0),.95)) if active.any() else 0.
    bands={name:dict(gain=float(point[j]),conditional_bootstrap_se=float(se[j]),
                    simultaneous_ci95=[float(point[j]-critical*se[j]),float(point[j]+critical*se[j])])
           for j,name in enumerate(pair_names)}
    r.save_json(HERE/'simultaneous-bands.json',dict(comparison_count=len(pairs),critical_value=critical,bands=bands,
        caveat='Conditional on frozen predictions; current-round multiplicity only. Does not account for historical selection or model refitting.'))
    for filename,records in [('comparison.csv',rows),('fold-metrics.csv',foldrows),('inner-weight-uncertainty.csv',selection)]:
        fields=list(dict.fromkeys(k for record in records for k in record))
        with (HERE/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    with (HERE/'oof-errors.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','fold','target','noise',*predictions])
        for i in range(400):w.writerow([data['ids'][dev[i]],fold[0,i],y[i],data['noise'][dev[i]],*[p[i]-y[i] for p in predictions.values()]])
    groups={'noise<12':data['noise'][dev]<12,'noise>=12':data['noise'][dev]>=12}
    cuts=np.quantile(y,[.25,.5,.75])
    for j in range(4):groups[f'target_quartile{j+1}']=np.digitize(y,cuts)==j
    group_metrics={name:{tag:dict(n=int(mask.sum()),rmse=r.rmse(p[mask],y[mask]),bias=float(np.mean(p[mask]-y[mask])))
                         for tag,mask in groups.items()} for name,p in predictions.items()}
    r.save_json(HERE/'results.json',dict(rows=rows,folds=foldrows,comparisons=comparisons,controls=cross,
                                       inner_weights=selection,group_metrics=group_metrics,
                                       noise_calibration=noise_reports,
                                       simultaneous_bands=bands,
                                       evaluated_images=400,evaluated_outer_folds=5,partition='original',
                                       weights_selected_on_outer_labels=False,submitted=False))
    text=['# 073 전체 구성: 내부 가중치 선택을 적용한 CNN 검증','',
          '기준은 원래400개·5-fold의 수정된073이다. 3개 partition을 합친 기존9.58169와 이 표의 단일 partition 점수를 직접 비교하면 안 된다.',
          f'기준 RMSE: **{r.rmse(base,y):.8f}**. 외부 예측은 기존 구성 요소를 그대로 사용하고, 증강 원본은 각 학습 fold 안으로 제한했다.','',
          '정상 샘플에서 `073 = .16 × 021 + .84 × T`이다. 추가는 `073 + λr`, 대체는 `T + λr`이며, λ는 내부B107에서만 선택했다. 분할 실패2개는 모든 방식에서 기존021 예측을 유지했다.','',
          '| 모델 | 방식 | RMSE | 073 대비 개선 | 대응 대조군 대비 보정 효과 |','|---|---|---:|---:|---:|']
    for row in rows:
        gain=row.get('gain_vs_matched_control');text.append(f"| {row['model']} | {row['arm']} | {row['rmse']:.6f} | {row['gain_vs073']:+.6f} | {'—' if gain is None else f'{gain:+.6f}'} |")
    text+=['','대체 방식의 대응 대조군은021을 제거한T이다. 공동 가중치 방식은 내부에서021 비중만 정한 대조군과 비교한다. 이 비교가 음수라면 전체073보다 좋아도 CNN이 개선을 만들었다고 해석할 수 없다.','',
           '| 모델·방식 | fold0 | fold1 | fold2 | fold3 | fold4 |','|---|---:|---:|---:|---:|---:|']
    for name in predictions:
        vals=[row['rmse'] for row in foldrows if row['model']==name]
        text.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    text+=['','95% 구간은 저장된 외부 예측에 대한 ID bootstrap 조건부 구간이다. 이번 비교군 안의 다중 비교를 고려한 동시 구간도 `simultaneous-bands.json`에 별도로 계산했다. 기존에 탐색한400개 샘플이며, 과거 연구 선택 편향과 재학습 변동을 모두 포함하지 않는다. 다른seed도 같은 데이터이므로 독립 검증셋으로 세지 않는다.','',
           '확실한 채택 여부는 `results.json`의 대조군 비교와 반복seed, fold별 방향, 상위오차 사례 의존성을 함께 검토해야 한다. 자동으로 기존061/073을 교체하거나 제출하지 않는다.','',
           '재현 명령은 [RUN.md](RUN.md), 사전 계획은 [PLAN.md](PLAN.md)와 [AMENDMENT-01.md](AMENDMENT-01.md), 전체 수치와 대조군 비교는 [results.json](results.json)에 있다. 각 모델 디렉터리의 `verification.json`에 누수·재로딩·정답 변조 검증이 기록된다.']
    (HERE/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
