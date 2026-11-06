"""Matched inner-only calibration; never fit weights to outer OOF labels."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import json
import numpy as np
import study as s


def predictions(arm, repeats, k, stage):
    if arm == 'old':
        paths = [s.R4/f'aux-seed{j}/fold{k}-{stage}/predictions.npz' for j in repeats]
    else:
        paths = [s.HERE/f'models/{arm}-seed{j}/fold{k}-{stage}/predictions.npz' for j in repeats]
    zs = [np.load(_repo_io(p)) for p in paths]
    for z in zs[1:]:
        assert np.array_equal(z['validation_index'], zs[0]['validation_index'])
        assert np.array_equal(z['train_index'], zs[0]['train_index'])
    return zs[0]['validation_index'], np.mean([z['validation_prediction'] for z in zs], axis=0)


def calibrate(direction, target):
    denominator = float(direction@direction)
    raw = float(direction@target/denominator) if denominator > 1e-12 else 0.
    return float(np.clip(raw, 0, 1)), raw


def evaluate(arm, repeats, d):
    n = len(d['dev']); base = np.load(_repo_io(s.R4.parent/'scope.npz'))['strict_blend'][0]
    pred = base.copy(); raw_oof = np.full(n, np.nan); weights = []
    labels = d['y']; pos = {int(q): i for i, q in enumerate(d['dev'])}
    for k in range(5):
        inner = np.load(_repo_io(s.R4/f'aux-seed0/fold{k}-selection.npz'))
        B, rb = predictions(arm, repeats, k, 'selector')
        V, rv = predictions(arm, repeats, k, 'final')
        assert np.array_equal(B, inner['B']) and not set(B)&set(V)
        good = inner['eligible'] & (d['noise'][B]>=12)
        coefficient, unconstrained = calibrate(rb[good], (labels[B]-inner['base073'])[good])
        poisoned = labels.copy(); poisoned[V] += 10000
        assert calibrate(rb[good], (poisoned[B]-inner['base073'])[good]) == (coefficient, unconstrained)
        ix = np.array([pos[int(q)] for q in V])
        assert np.all(d['fold'][0,ix] == k)
        eligible = (d['noise'][V]>=12) & ~d['fail'][0,ix]
        pred[ix[eligible]] += coefficient*rv[eligible]
        raw_oof[ix] = rv
        weights.append(dict(fold=k, coefficient=coefficient, unconstrained=unconstrained,
                            n_calibration=int(good.sum()), calibration_ids=d['ids'][B[good]].tolist(),
                            outer_ids=d['ids'][V].tolist(), outer_poison_independent=True,
                            correction_rms_calibration=float(np.sqrt(np.mean(rb[good]**2)))))
    assert np.isfinite(raw_oof).all() and np.isfinite(pred).all()
    return pred, raw_oof, weights


def contrast(base, pred, y, folds, bootstrap):
    a = (base-y)**2; b = (pred-y)**2
    draws = np.sqrt(a[bootstrap].mean(1))-np.sqrt(b[bootstrap].mean(1))
    ga = np.sqrt(a.mean())-np.sqrt(b.mean())
    fg = [float(np.sqrt(a[folds==k].mean())-np.sqrt(b[folds==k].mean())) for k in range(5)]
    order = np.argsort(a-b)[::-1]; remaining = order[10:]
    return dict(gain=float(ga), ci95=np.quantile(draws,[.025,.975]).tolist(),
                better_fold_count=int(np.sum(np.array(fg)>1e-10)), fold_gains=fg,
                gain_without_largest10_beneficiaries=float(np.sqrt(a[remaining].mean())-np.sqrt(b[remaining].mean())),
                largest10_net_gain_share=float((a-b)[order[:10]].sum()/(a-b).sum()) if (a-b).sum()!=0 else None,
                error_correlation=float(np.corrcoef(base-y,pred-y)[0,1]))


def main(repeat_arms):
    s.protect()
    completed = json.loads((s.HERE/'training-seed0-complete.json').read_text())
    assert set(completed['arms']) == set(s.ARMS) and completed['complete']
    d = dict(np.load(_repo_io(s.R4.parent/'inputs.npz'))); y=d['y'][d['dev']]; fold=d['fold'][0]
    primary = np.load(_repo_io(s.R4.parent/'scope.npz'))['strict_blend'][0]
    frozen = np.load(_repo_io(s.R4/'noise-calibration/aux-oof.npz'))['addition']
    bootstrap = np.random.default_rng(2026101617).integers(0,len(y),(10000,len(y)))
    plans = [('old-seed0','old',[0]), ('old-mean2','old',[0,1])]
    plans += [(arm+'-seed0',arm,[0]) for arm in s.ARMS]
    for arm in repeat_arms:
        plans += [(arm+'-seed1',arm,[1]),(arm+'-mean2',arm,[0,1])]
    preds = {'strict073':primary, 'old-frozen-noise-mean2':frozen}; raw={}; selections={}
    for tag,arm,repeats in plans:
        pred,rv,ww=evaluate(arm,repeats,d)
        preds[tag]=pred;raw[tag]=rv;selections[tag]=ww
    reports={}; p76=np.load(_repo_path(s.ROOT, 'experiments/push-1300/dev-combo.npy'))[0]
    gate=(d['noise'][d['dev']]>=12)&~d['fail'][0]
    for name,p in preds.items():
        record=s.metrics(y,p,fold)
        record.update(high_rmse=float(np.sqrt(np.mean((p[gate]-y[gate])**2))),
                      low_rmse=float(np.sqrt(np.mean((p[~gate]-y[~gate])**2))),
                      vs_strict073=contrast(primary,p,y,fold,bootstrap),
                      vs_old_seed0_matched=contrast(preds['old-seed0'],p,y,fold,bootstrap),
                      vs_old_mean2_matched=contrast(preds['old-mean2'],p,y,fold,bootstrap),
                      vs_old_frozen_mean2=contrast(frozen,p,y,fold,bootstrap),
                      transported_to_recorded076=s.metrics(y,p76+p-primary,fold))
        if name in raw:
            record['raw_correction_correlation_old_seed0_high']=float(np.corrcoef(raw[name][gate],raw['old-seed0'][gate])[0,1])
        reports[name]=record
    advance=[]
    for arm in s.ARMS:
        check=reports[arm+'-seed0']['vs_old_seed0_matched']
        if check['gain']>=.02 and check['better_fold_count']>=3: advance.append(arm)
    # Conditional simultaneous RMSE-gain bands for the new comparisons only.
    names=[n for n in preds if n.startswith(s.ARMS)]
    base_sq=(preds['old-seed0']-y)**2
    draws=np.array([np.sqrt(base_sq[bootstrap].mean(1))-np.sqrt(((preds[n]-y)**2)[bootstrap].mean(1)) for n in names])
    gains=np.array([reports[n]['vs_old_seed0_matched']['gain'] for n in names])
    sd=np.std(draws,axis=1,ddof=1); safe=np.maximum(sd,1e-12)
    critical=float(np.quantile(np.max(np.abs((draws-gains[:,None])/safe[:,None]),axis=0),.95))
    bands={n:[float(g-critical*t),float(g+critical*t)] for n,g,t in zip(names,gains,sd)}
    result=dict(results=reports, selections=selections, advance_by_locked_rule=advance,
                simultaneous_new_comparisons=dict(critical=critical,bands=bands,names=names),
                limitations=['Repeated use of400 labels; conditional ID bootstrap ignores training-set dependence.',
                             'Historical076 transport is not a fresh strict-source076 model fit.',
                             'Matched control isolates training from the calibration-rule change.',
                             'No Public coefficient fitting or fresh independent holdout.'])
    suffix='confirmation' if repeat_arms else 'pilot'
    s.save(_repo_io(f'{suffix}-results.json'),result)
    np.savez_compressed(_repo_io(s.HERE/f'{suffix}-oof.npz'),ids=d['ids'][d['dev']],y=y,fold=fold,**preds)
    with (s.HERE/f'{suffix}-comparison.csv').open('w',newline='') as f:
        import csv
        w=csv.writer(f);w.writerow(['arm','rmse','gain_vs_old_seed0','gain_vs_old_frozen_mean2','fold0','fold1','fold2','fold3','fold4'])
        for n,rr in reports.items():w.writerow([n,rr['rmse'],rr['vs_old_seed0_matched']['gain'],rr['vs_old_frozen_mean2']['gain'],*rr['folds']])
    s.protect()
    print(json.dumps(dict(scores={n:rr['rmse'] for n,rr in reports.items()},advance=advance,bands=bands),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repeat-arms',nargs='*',choices=s.ARMS,default=[])
    main(parser.parse_args().repeat_arms)
