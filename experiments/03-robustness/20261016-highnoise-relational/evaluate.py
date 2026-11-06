"""Complete-fold evaluation of the predeclared nested calibration policy."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import experiment as e
import numpy as np,json,sys,argparse,csv

def weight(pred,base,labels,eligible):
    d=(pred-base)[eligible];r=(labels-base)[eligible];den=float(d@d)
    return float(np.clip(d@r/den,0,.25)) if den>1e-12 else 0.

def metrics(y,b,p,fold,draw):
    a=(b-y)**2;c=(p-y)**2;gain=e.rmse(b,y)-e.rmse(p,y);gg=np.sqrt(a[draw].mean(1))-np.sqrt(c[draw].mean(1))
    order=np.argsort(-(a-c));D=np.c_[np.ones(len(b)),b-b.mean()];coef=np.linalg.lstsq(D,p-b,rcond=None)[0];orth=(p-b)-D@coef
    return dict(baseline_rmse=e.rmse(b,y),candidate_rmse=e.rmse(p,y),gain=gain,conditional_ci95=np.quantile(gg,[.025,.975]).tolist(),
        two_comparison_bonferroni_ci=np.quantile(gg,[.0125,.9875]).tolist(),
        folds=[dict(fold=k,baseline_rmse=e.rmse(b[fold==k],y[fold==k]),candidate_rmse=e.rmse(p[fold==k],y[fold==k]),gain=e.rmse(b[fold==k],y[fold==k])-e.rmse(p[fold==k],y[fold==k])) for k in range(5)],
        improved_folds=int(sum(e.rmse(p[fold==k],y[fold==k])<e.rmse(b[fold==k],y[fold==k]) for k in range(5))),
        removed_largest_gain={str(n):dict(gain=e.rmse(b[order[n:]],y[order[n:]])-e.rmse(p[order[n:]],y[order[n:]]),share_net_SSE_gain=float((a-c)[order[:n]].sum()/(a-c).sum()) if abs((a-c).sum())>1e-10 else None) for n in [5,10,20]},
        n_improved=int(np.sum(c<a-1e-10)),n_harmed=int(np.sum(c>a+1e-10)),change_rms=float(np.sqrt(np.mean((p-b)**2))),
        affine_decomposition=dict(intercept=float(coef[0]),slope_delta=float(coef[1]),affine_only_gain=e.rmse(b,y)-e.rmse(b+D@coef,y),orthogonal_change_gain=e.rmse(b,y)-e.rmse(b+orth,y),diagnostic_only=True))

def main(repeats):
    for repeat in repeats:assert json.loads((e.HERE/f'training-complete-seed{repeat}.json').read_text())['complete']
    d=dict(np.load(_repo_io(e.HERE/'inputs.npz')));graphs=json.loads((e.HERE/'graphs.json').read_text());dev=d['dev'];y=d['y'][dev];fold=d['fold'];noise=d['noise'][dev];base=d['base082']
    ps={a:base.copy() for a in e.ARMS};fixedA={a:base.copy() for a in e.ARMS};direct={a:np.zeros(400) for a in e.ARMS};records=[]
    for k,g in enumerate(graphs):
        z=np.load(_repo_io(e.HERE/f'anchors/fold{k}.npz'));A,B,T,V=[z[n] for n in ['A','B','T','V']];ix=z['ix']
        row=dict(fold=k,n_calibration_high=int(z['highB'].sum()),n_validation_high=int(z['highV'].sum()),old_weight=float(z['old_weight']),arms={})
        for arm in e.ARMS:
            selects=[];refits=[]
            for repeat in repeats:
                a=np.load(_repo_io(e.HERE/f'models/seed{repeat}/fold{k}-selector-{arm}/predictions.npz'));b=np.load(_repo_io(e.HERE/f'models/seed{repeat}/fold{k}-refit-{arm}/predictions.npz'))
                assert np.array_equal(a['train'],A) and np.array_equal(a['query'],np.r_[B,V]) and np.array_equal(b['train'],T) and np.array_equal(b['query'],V)
                selects.append(a['prediction']);refits.append(b['prediction'])
            s=np.mean(selects,axis=0);aB=s[:len(B)];aAV=s[len(B):];aV=np.mean(refits,axis=0)
            lam=weight(aB,z['p82B'],d['y'][B].copy(),z['highB']);poison=d['y'].copy();poison[V]+=10000
            assert lam==weight(aB,z['p82B'],poison[B].copy(),z['highB'])
            ps[arm][ix]=base[ix]+z['highV']*lam*(aV-base[ix]);fixedA[arm][ix]=base[ix]+z['highV']*lam*(aAV-base[ix]);direct[arm][ix]=aV
            ib=np.flatnonzero(z['highB']);rng=np.random.default_rng(2026101660+k);resamples=rng.integers(0,len(ib),(2000,len(ib)));dd=(aB-z['p82B'])[ib];rr=(d['y'][B]-z['p82B'])[ib]
            den=(dd[resamples]**2).sum(1);boot=np.clip((dd[resamples]*rr[resamples]).sum(1)/np.maximum(den,1e-12),0,.25)
            row['arms'][arm]=dict(weight=lam,conditional_coefficient_ci95=np.quantile(boot,[.025,.975]).tolist(),coefficient_uncertainty_scope='Resamples B noisy rows with the already-selected old082 coefficient held fixed; excludes encoder refitting and old-weight selection uncertainty.',
                selector_A213_B107_image_rmse=e.rmse(aB,d['y'][B]),selector_A213_V80_image_rmse=e.rmse(aAV,d['y'][V]),refit_T320_V80_image_rmse=e.rmse(aV,d['y'][V]),
                student_refit_shift_rms=float(np.sqrt(np.mean((aV-aAV)**2))),outer_label_poison_coefficient_exact=True)
        records.append(row)
    draw=np.random.default_rng(2026101652).integers(0,400,(10000,400));results={}
    for arm,p in ps.items():
        r=metrics(y,base,p,fold,draw);r['versus076']=metrics(y,d['base076'],p,fold,draw);r['fixed_A_student']=metrics(y,base,fixedA[arm],fold,draw)
        r['image_only_rmse']=e.rmse(direct[arm],y);high=noise>=12;r['high_noise_image_only_rmse']=e.rmse(direct[arm][high],y[high]);r['high_noise_ensemble_rmse']=e.rmse(p[high],y[high]);r['baseline_high_noise_rmse']=e.rmse(base[high],y[high])
        r['high_noise_error_correlation']=float(np.corrcoef((direct[arm]-y)[high],(base-y)[high])[0,1]);r['low_rows_unchanged']=bool(np.array_equal(p[~high],base[~high]));assert r['low_rows_unchanged']
        results[arm]=r
    matched=metrics(y,ps['mse'],ps['relational'],fold,draw);shuffled=metrics(y,ps['shuffled'],ps['relational'],fold,draw);r=results['relational']
    checks=dict(overall_gain_at_least_point03=r['gain']>=.03,three_folds=r['improved_folds']>=3,baseline_ci_lower_positive=r['conditional_ci95'][0]>0,matched_MSE_ci_lower_positive=matched['conditional_ci95'][0]>0,ten_removed_gain_positive=r['removed_largest_gain']['10']['gain']>0,beats_shuffled=shuffled['gain']>0)
    tag=''.join(map(str,repeats));e.save(_repo_io(f'results-{tag}.json'),dict(results=results,relational_vs_mse=matched,relational_vs_shuffled=shuffled,selection=records,
        limitations=['All labels previously exposed. Conditional paired-ID uncertainty is not independent private-leaderboard evidence.','New pixel models and both calibration stages exclude V; inherited082 representations are frozen, not fully cross-fitted afresh.','Fixed-A comparison changes only the new student; inherited baseline still uses its original outer refit. It is not a matched-size comparison of the complete pipeline.']))
    e.save(_repo_io(f'decision-{tag}.json'),dict(advance=all(checks.values()),checks=checks,repeats=repeats,incumbent='082',rollback='076',official_promotion=False,new_submission=False))
    e.arrays(f'oof-{tag}.npz',ids=d['ids'][dev],y=y,fold=fold,noise=noise,baseline076=d['base076'],baseline082=base,**ps,**{a+'_image':p for a,p in direct.items()},**{a+'_fixedA':p for a,p in fixedA.items()})
    with (e.HERE/f'comparison-{tag}.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','RMSE','gain_vs082','CI95_low','CI95_high','improved_folds','high_noise_RMSE','gain_after_top10_removed'])
        for a,r in results.items():w.writerow([a,r['candidate_rmse'],r['gain'],*r['conditional_ci95'],r['improved_folds'],r['high_noise_ensemble_rmse'],r['removed_largest_gain']['10']['gain']])
    with (e.HERE/f'per-image-{tag}.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','fold','noise','hardness','082','076',*e.ARMS,*[a+'_SSE_gain' for a in e.ARMS]])
        for i in range(400):w.writerow([d['ids'][dev[i]],fold[i],noise[i],y[i],base[i],d['base076'][i],*[ps[a][i] for a in e.ARMS],*[(base[i]-y[i])**2-(ps[a][i]-y[i])**2 for a in e.ARMS]])
    print((e.HERE/f'comparison-{tag}.csv').read_text());print('REL vs MSE',json.dumps(matched));print('ADVANCE',all(checks.values()),checks,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repeats',type=int,nargs='+',default=[0]);args=ap.parse_args();assert args.repeats in [[0],[0,1]]
    sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main(args.repeats)
