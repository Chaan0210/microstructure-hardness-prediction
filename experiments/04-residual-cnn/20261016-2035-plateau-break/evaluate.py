"""Score the complete predeclared screens once, retaining unfavorable results."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study as s
import csv,json,sys
import numpy as np

def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))

def compare(y,b,p,draw,fold):
    a=(b-y)**2;c=(p-y)**2;delta=a-c;gain=rmse(y,b)-rmse(y,p)
    boots=np.sqrt(a[draw].mean(1))-np.sqrt(c[draw].mean(1))
    order=np.argsort(-delta)
    design=np.c_[np.ones(len(b)),b-b.mean()];coef=np.linalg.lstsq(design,p-b,rcond=None)[0]
    orth=(p-b)-design@coef
    return dict(rmse=rmse(y,p),baseline_rmse=rmse(y,b),gain=gain,
        conditional_ci95=np.percentile(boots,[2.5,97.5]).tolist(),
        conditional_bonferroni_three_ci=np.percentile(boots,[100*.05/6,100*(1-.05/6)]).tolist(),
        folds=[dict(fold=int(k),baseline=rmse(y[fold==k],b[fold==k]),candidate=rmse(y[fold==k],p[fold==k]),gain=rmse(y[fold==k],b[fold==k])-rmse(y[fold==k],p[fold==k])) for k in np.unique(fold)],
        improved_folds=int(sum(rmse(y[fold==k],p[fold==k])<rmse(y[fold==k],b[fold==k]) for k in np.unique(fold))),
        n_improved=int(np.sum(delta>1e-10)),n_harmed=int(np.sum(delta< -1e-10)),
        removed_top_gain={str(n):dict(gain=rmse(y[order[n:]],b[order[n:]])-rmse(y[order[n:]],p[order[n:]]),
            share_net_sse_gain=float(delta[order[:n]].sum()/delta.sum()) if abs(delta.sum())>1e-9 else None) for n in [5,10,20]},
        change_rms=float(np.sqrt(np.mean((p-b)**2))),change_mean=float(np.mean(p-b)),
        change_affine_projection=dict(intercept=float(coef[0]),slope_delta=float(coef[1]),
            affine_only_gain=rmse(y,b)-rmse(y,b+design@coef),orthogonal_change_gain=rmse(y,b)-rmse(y,b+orth),
            status='Descriptive decomposition on fixed OOF predictions, not a fitted or selectable candidate'))

def main():
    d,yall=s.load_data();dev=d['dev'];y=yall[dev];z=d['base'];b=z['p76'];fold=z['fold'];noise=z['noise'];fail=z['failed'];ids=d['ids'][dev]
    rebuilt=.16*z['G']+.384*z['tab076']+.128*z['ridge076']+.128*z['logridge076']+.05*d['C73_set'][0]+.15*d['C73_gmm'][0]
    rebuilt[fail]=z['G'][fail];assert np.max(abs(rebuilt-b))<1e-10
    predictions={};primary={}
    if (s.HERE/'h1-oof.npz').exists():
        h1=np.load(_repo_io(s.HERE/'h1-oof.npz'));assert np.array_equal(h1['ids'],ids)
        for key in ['raw','corrected']:predictions['H1_'+key]=b+.15*(h1[key]-d['C73_gmm'][0])
        primary['H1_corrected']='H1_raw'
    for number,keys,w,reference in [(2,['linear','marginal','joint'],.15,d['C73_gmm'][0]),(3,['identity','invariant','rotated'],.128,z['ridge076'])]:
        h=np.load(_repo_io(s.HERE/f'h{number}-oof.npz'));assert np.array_equal(h['ids'],ids) and np.array_equal(h['fold'],fold)
        for key in keys:predictions[f'H{number}_{key}']=b+w*(h[key]-reference)
    primary.update(H2_joint='H2_marginal',H3_invariant='H3_identity')
    for k in predictions:predictions[k]=np.where(fail,b,predictions[k]);assert np.isfinite(predictions[k]).all()
    draw=np.random.default_rng(2026101636).integers(0,len(y),(10000,len(y)))
    bands={'clean':noise<9.5,'middle':(noise>=9.5)&(noise<12),'high':noise>=12}
    current=np.load(_repo_path(s.ROOT, 'experiments/rmse-research-20261015/round4-full073/noise-calibration/aux-oof.npz'))
    cnn_delta=current['addition']-current['baseline'];current_b=b+cnn_delta
    results={};decisions={}
    for name,p in predictions.items():
        r=compare(y,b,p,draw,fold)
        r['bands']={k:dict(n=int(m.sum()),baseline_rmse=rmse(y[m],b[m]),candidate_rmse=rmse(y[m],p[m]),gain=rmse(y[m],b[m])-rmse(y[m],p[m])) for k,m in bands.items()}
        r['on_existing_CNN']=compare(y,current_b,p+cnn_delta,draw,fold)
        if name in primary:
            r['matched_control']=compare(y,predictions[primary[name]],p,draw,fold)
            mechanisms=r['matched_control']['gain']>0
            if name=='H2_joint':mechanisms &= rmse(y,p)<rmse(y,predictions['H2_linear'])
            if name=='H3_invariant':mechanisms &= rmse(y,p)<rmse(y,predictions['H3_rotated'])
            checks=dict(gain_at_least_point03=r['gain']>=.03,three_folds=r['improved_folds']>=3,
                conditional_lower_positive=r['conditional_ci95'][0]>0,ten_removed_positive=r['removed_top_gain']['10']['gain']>0,
                mechanism_controls=bool(mechanisms))
            decisions[name]=dict(advance=all(checks.values()),checks=checks)
        results[name]=r
    diag=dict(n=len(y),baseline_rmse=rmse(y,b),existing_CNN_rmse=rmse(y,current_b),source_restricted_reassembly_max_diff=float(np.max(abs(rebuilt-b))),
        bands={k:dict(n=int(m.sum()),rmse=rmse(y[m],b[m]),sse_share=float(np.sum((y[m]-b[m])**2)/np.sum((y-b)**2)),bias=float(np.mean(b[m]-y[m]))) for k,m in bands.items()},
        top10_baseline_sse_share=float(np.sort((y-b)**2)[-10:].sum()/np.sum((y-b)**2)),
        residual_correlations={k:float(np.corrcoef((b-y)[noise>=12],(d[c][0]-y)[noise>=12])[0,1]) for k,c in [('G021','C73_cnn'),('GMM','C73_gmm'),('grainset','C73_set')]},
        limitations=['All500 labels previously exposed; no untouched holdout.',
            'Frozen U-Net/Cellpose and historical feature/ensemble choices are not fully nested.',
            'Intervals condition on fitted OOF predictions; they omit repeated adaptive research and overlapping training-set dependence.',
            'Production mean CNN coefficient has no separate honest OOF score; sensitivity uses original inner-fold coefficients.'])
    s.save(_repo_io('baseline-diagnosis.json'),diag);s.save(_repo_io('results.json'),results);s.save(_repo_io('decision.json'),dict(incumbent='076',promoted=False,decisions=decisions,advance=[k for k,v in decisions.items() if v['advance']]))
    s.arrays('all-predictions.npz',ids=ids,y=y,fold=fold,noise=noise,baseline=b,baseline_plus_existing_CNN=current_b,**predictions)
    with (s.HERE/'comparison.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','rmse','gain','ci_low','ci_high','folds_improved','gain_after_top10_removed','matched_control_gain','advance'])
        for k,r in results.items():w.writerow([k,r['rmse'],r['gain'],*r['conditional_ci95'],r['improved_folds'],r['removed_top_gain']['10']['gain'],r.get('matched_control',{}).get('gain'),decisions.get(k,{}).get('advance')])
    with (s.HERE/'per-image.csv').open('x',newline='') as f:
        names=list(predictions);w=csv.writer(f);w.writerow(['ID','fold','noise','y','baseline','baseline_error',*names,*[n+'_sse_gain' for n in names]])
        for i in range(len(y)):w.writerow([ids[i],fold[i],noise[i],y[i],b[i],b[i]-y[i],*[predictions[n][i] for n in names],*[(b[i]-y[i])**2-(predictions[n][i]-y[i])**2 for n in names]])
    print((s.HERE/'comparison.csv').read_text());print(json.dumps(decisions,indent=2))

if __name__=='__main__':
    sys.addaudithook(s.guard)
    with s.threadpool_limits(limits=1):main()
