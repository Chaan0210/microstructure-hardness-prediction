"""Repair H2's tied-value ordering defect without changing its search protocol."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study as s
import evaluate as e
import verify as v
import numpy as np
import json,sys
PARENT=s.HERE

def empirical_quantile(x,w,quantiles):
    # Sorting by weight within ties fixes floating summation order as well.
    ix=np.lexsort((w,x));cw=np.cumsum(w[ix]);j=np.searchsorted(cw,np.asarray(quantiles)*cw[-1],side='left')
    return x[ix[np.minimum(j,len(ix)-1)]]

def main():
    out=PARENT/'h2-tie-repair';out.mkdir(exist_ok=False)
    plan=(PARENT/'PLAN.md').read_text()+'''\n\n## Implementation repair before re-evaluation\nThe original interpolated weighted quantile failed a tied-value permutation test\n(max difference0.05519 on a deterministic tied-value fixture). Real float16 maps\ncontain ties, so this violates the intended distribution representation. Replace\nit by the inverse weighted empirical CDF with canonical ordering within ties.\nNo hypothesis, feature directions, seed, split, kernel grid, comparison, selection\nprocedure or stopping criterion changes. Preserve the original results and report\nthe repaired run as the final H2 test. No further quantile variant is searched.\n'''
    (out/'PLAN.md').write_text(plan);(out/'incumbent.json').write_text((PARENT/'incumbent.json').read_text())
    source=dict(original_study_sha256=s.digest(PARENT/'study.py'),repair_sha256=s.digest(__file__),plan_sha256=s.digest(out/'PLAN.md'))
    (out/'pre-score-lock.json').write_text(json.dumps(source,indent=2))
    rng=np.random.default_rng(39);x=np.repeat(np.arange(5,dtype=float),20);w=rng.uniform(.1,1,100);p=rng.permutation(100);q=np.linspace(.025,.975,17)
    old=float(np.max(abs(s.weighted_quantile(x,w,q)-s.weighted_quantile(x[p],w[p],q))))
    assert old>.05
    assert np.array_equal(empirical_quantile(x,w,q),empirical_quantile(x[p],w[p],q))
    # Independent literal weighted-CDF inversion, including tied observations.
    expected=np.array([min(a for a in np.unique(x) if w[x<=a].sum()>=t*w.sum()) for t in q])
    assert np.array_equal(empirical_quantile(x,w,q),expected)
    s.HERE=out;s.weighted_quantile=empirical_quantile
    sys.addaudithook(s.guard)
    s.save(_repo_io('quantile-check.json'),dict(original_tie_permutation_error=old,repaired_tie_permutation_error=0.,independent_weighted_CDF_exact=True,source=source))
    s.extract();s.run_h2()
    d,yall=s.load_data();dev=d['dev'];y=yall[dev];z=d['base'];b=z['p76'];fold=z['fold'];m=np.load(_repo_io(out/'measurements.npz'));h=np.load(_repo_io(out/'h2-oof.npz'));sig=m['signature']
    draw=np.random.default_rng(2026101636).integers(0,400,(10000,400));pred={k:np.where(z['failed'],b,b+.15*(h[k]-d['C73_gmm'][0])) for k in ['linear','marginal','joint']}
    results={k:e.compare(y,b,p,draw,fold) for k,p in pred.items()}
    results['joint']['matched_control']=e.compare(y,pred['marginal'],pred['joint'],draw,fold)
    current=np.load(_repo_path(s.ROOT, 'experiments/rmse-research-20261015/round4-full073/noise-calibration/aux-oof.npz'));delta=current['addition']-current['baseline']
    results['joint']['on_existing_CNN']=e.compare(y,b+delta,pred['joint']+delta,draw,fold)
    r=results['joint'];checks=dict(gain_at_least_point03=r['gain']>=.03,three_folds=r['improved_folds']>=3,conditional_lower_positive=r['conditional_ci95'][0]>0,
        ten_removed_positive=r['removed_top_gain']['10']['gain']>0,mechanism_controls=r['matched_control']['gain']>0 and e.rmse(y,pred['joint'])<e.rmse(y,pred['linear']))
    selections=json.loads((out/'h2-selection.json').read_text())['records'];models=[]
    for rec in selections:
        if rec['kind']!='joint':continue
        k=rec['fold'];noisy=rec['noisy'];tr=dev[fold!=k];ix=np.flatnonzero((fold==k)&((d['noise'][dev]>=12)==noisy));query=dev[ix]
        obj=v.model_bundle(s.gmm_columns(d,tr,noisy),yall[tr].copy(),'joint',rec['alpha'],sig[tr],rec['extra_scale']);obj.update(train_ids=d['ids'][tr],query_ids=d['ids'][query])
        path=out/f'model-fold{k}-gate{int(noisy)}.joblib';s.joblib.dump(obj,_repo_io(path));reloaded=v.predict(s.joblib.load(_repo_io(path)),s.gmm_columns(d,query,noisy),sig[query]);err=float(np.max(abs(reloaded-h['joint'][ix])));assert err<1e-8
        models.append(dict(fold=k,noisy=noisy,reload_max_error=err))
    tr=dev[fold!=0];ix=np.flatnonzero((fold==0)&(d['noise'][dev]<12));query=dev[ix];poison=yall.copy();poison[np.setdiff1d(np.arange(500),tr)]+=10000
    pp,_=s.select_h2(s.gmm_columns(d,tr,False),sig[tr],poison[tr].copy(),s.gmm_columns(d,query,False),sig[query],2026101635,'joint');assert np.array_equal(pp,h['joint'][ix])
    original=np.load(_repo_io(PARENT/'h2-oof.npz'));assert np.array_equal(h['linear'],original['linear'])
    # Real-map ties and spatial permutations: the repaired signature is exact.
    raw=np.load(_repo_path(s.ROOT, 'experiments/push-400/unet4maps-z1-train.npz'))['maps'][0].astype(float);pr=raw[:4];pr/=pr.sum(0);g=np.maximum(pr[0]+pr[1],1e-6)
    physical=np.stack([(raw[4]-5.636784553527832)/.9691801071166992,(raw[7]-.15)/.15,np.hypot(raw[5],raw[6])/.3,2*pr[1]/g],-1).reshape(-1,4);dirs=m['directions'];proj=physical@dirs.T;weights=g.ravel();perm=np.random.default_rng(2026101639).permutation(len(weights))
    alt=np.array([empirical_quantile(proj[perm,j],weights[perm],m['quantiles']) for j in range(20)])
    assert np.array_equal(alt,sig[0])
    s.arrays('evaluated-predictions.npz',ids=d['ids'][dev],y=y,fold=fold,baseline=b,**pred)
    with (out/'per-image.csv').open('x',newline='') as f:
        wr=s.csv.writer(f);wr.writerow(['ID','fold','target','baseline','linear','marginal','joint','joint_SSE_gain'])
        for j,i in enumerate(d['ids'][dev]):wr.writerow([i,fold[j],y[j],b[j],pred['linear'][j],pred['marginal'][j],pred['joint'][j],(b[j]-y[j])**2-(pred['joint'][j]-y[j])**2])
    s.save(_repo_io('results.json'),results);s.save(_repo_io('decision.json'),dict(advance=all(checks.values()),checks=checks));s.save(_repo_io('verification.json'),dict(all_passed=True,models=models,real_map_permutation_exact=True,heldout_label_poison_exact=True,linear_control_unchanged=True,source=source))
    print('H2 REPAIRED',json.dumps({k:dict(rmse=r['rmse'],gain=r['gain'],ci=r['conditional_ci95']) for k,r in results.items()}),'advance',all(checks.values()),flush=True)

if __name__=='__main__':
    with s.threadpool_limits(limits=1):main()
