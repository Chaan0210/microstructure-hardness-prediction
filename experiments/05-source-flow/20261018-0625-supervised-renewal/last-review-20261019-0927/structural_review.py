"""Three declared fixed block replacements, using matched training-only OOF."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json
import hashlib
import datetime
import numpy as np

OUT=Path(__file__).resolve().parent
HERE=OUT.parent
ROOT=_RepoPath(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    d=np.load(_repo_path(HERE, 'data.npz'));dev=d['dev'];ids=d['ids'][dev];y=d['y'][dev];fold=d['fold'];eligible=~d['failed']
    legacy_path=_repo_path(ROOT, 'experiments/076-complement-20261016/baseline076.npz')
    legacy=np.load(_repo_io(legacy_path));oldtab=np.load(_repo_path(HERE, 'legacy-tab-block-original400.npz'))
    for z in [legacy,oldtab]:
        assert np.array_equal(z['ids'],ids) and np.array_equal(z['fold'],fold)
    assert np.array_equal(legacy['p76'],d['base076']) and np.array_equal(legacy['failed'],d['failed'])
    expected_tab=legacy['tab076']+(d['base083']-d['base082'])/.384
    assert np.max(abs(expected_tab[eligible]-oldtab['prediction'][eligible]))<1e-10
    # Check only the exact existing high-noise linear contexts needed here.
    contexts=[]
    for k in range(5):
        path=legacy_path.parent/f'linear/fold{k}.npz';z=np.load(_repo_io(path));tr=dev[fold!=k]
        q=dev[(fold==k)&(legacy['noise']>=12)]
        assert np.array_equal(z['train'],tr) and np.array_equal(z['query'],q) and not set(tr)&set(q)
        ix=np.flatnonzero((fold==k)&(legacy['noise']>=12))
        assert np.array_equal(z['ridge'],legacy['ridge076'][ix]) and np.array_equal(z['logridge'],legacy['logridge076'][ix])
        contexts.append(dict(fold=k,train_rows=len(tr),query_rows=len(q),query_source_overlap=0,predictions_sha256=digest(path)))
    ref=np.load(_repo_path(HERE, 'source-h300/proper-summary-s1811-1823.npz'))
    for key,value in [('ids',ids),('y',y),('fold',fold)]:assert np.array_equal(ref[key],value)
    current=ref['flow38'];control_linear=np.load(_repo_path(HERE, 'flow-linear-check/oof.npz'))['control']
    per_seed=[];control_tab=np.full(400,np.nan);sources=[]
    for seed in [1811,1823]:
        values=np.full(400,np.nan)
        for k in range(5):
            folder=_repo_path(HERE, f'source-h300/proper-original400-f{k}-s{seed}')
            p=folder/'system38-predictions.npz';z=np.load(_repo_io(p));ix=np.flatnonzero(fold==k)
            r=json.loads((folder/'recipe.json').read_text())
            assert r['train']==dev[fold!=k].tolist() and r['query']==dev[ix].tolist()
            assert not set(r['source_indices'])&set(r['query'])
            assert np.array_equal(z['ids'],ids[ix]) and np.array_equal(z['y'],y[ix])
            rebuilt=z['base083']+eligible[ix]*(.384*(z['tab']-z['control'])+.256*(z['linear']-control_linear[ix]))
            assert np.max(abs(rebuilt-z['system']))<1e-10
            if seed==1811:control_tab[ix]=z['control']
            else:assert np.array_equal(control_tab[ix],z['control'])
            values[ix]=z['system'];sources.append(dict(path=str(p.relative_to(HERE)),sha256=digest(p)))
        assert np.isfinite(values).all();per_seed.append(values)
    assert np.array_equal(np.mean(per_seed,0),current)
    tab_delta=.384*eligible*(control_tab-oldtab['prediction'])
    linear_delta=.128*eligible*(2*control_linear-legacy['ridge076']-legacy['logridge076'])
    variants={'replace_tab_block':tab_delta,'replace_linear_block':linear_delta,'replace_both_blocks':tab_delta+linear_delta}
    draws=np.random.default_rng(2026101997).integers(400,size=(10000,400));old=(current-y)**2
    results={};predictions={}
    for name,delta in variants.items():
        p=current+delta;assert np.isfinite(p).all() and np.array_equal(p[~eligible],current[~eligible])
        new=(p-y)**2;boot=np.sqrt(old[draws].mean(1))-np.sqrt(new[draws].mean(1));keep=np.ones(400,bool)
        keep[np.argsort(old-new)[-10:]]=False
        folds=[float(np.sqrt(old[fold==k].mean())-np.sqrt(new[fold==k].mean())) for k in range(5)]
        seed_gains=[float(np.mean((v-y)**2)**.5-np.mean((v+delta-y)**2)**.5) for v in per_seed]
        result=dict(rmse=float(new.mean()**.5),gain=float(old.mean()**.5-new.mean()**.5),fold_gains=folds,
            positive_folds=sum(v>0 for v in folds),ci95=np.percentile(boot,[2.5,97.5]).tolist(),
            top10_positive_removed_gain=float(old[keep].mean()**.5-new[keep].mean()**.5),seed_gains=seed_gains,
            positive_images=int(np.sum(new<old)))
        checks=dict(gain_at_least_point_one=result['gain']>=.10,four_folds=result['positive_folds']>=4,
                    positive_interval=result['ci95'][0]>0,positive_influence=result['top10_positive_removed_gain']>0,
                    both_seed_gains_positive=all(v>0 for v in seed_gains))
        result.update(checks=checks,passes_declared_resource_gate=all(checks.values()));results[name]=result;predictions[name]=p
    record=dict(computed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference='User-submitted sourceF38; Public score not used in this calculation.',
        reference_rmse=float(old.mean()**.5),variants=results,rows=400,contexts=contexts,source_oof=sources,
        legacy076_sha256=digest(legacy_path),plan_sha256=digest(_repo_path(OUT, 'PLAN.txt')),all_rows_retained=True,
        test_inputs_read=False,weights_fitted=False,full_system_confirmation_passed=False,
        scope='Matched original400 conditional development only. New source encoders exclude validation images; inherited representations remain globally trained and all labels reused. Fixed inherited block masses; no Public score fit or test-data selection.')
    np.savez(_repo_path(OUT, 'structural-oof.npz'),ids=ids,y=y,fold=fold,reference=current,**predictions)
    (_repo_path(OUT, 'structural-results.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'reference_rmse':record['reference_rmse'],'variants':results},indent=2))


if __name__=='__main__':main()
