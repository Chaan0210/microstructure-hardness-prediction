"""Frozen, inner-only blend policy; evaluate once all five folds finish."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study as s
import numpy as np,json,csv,sys
ARMS=s.ARMS+['frozen_ssl','frozen_random']
def weight(pred,base,y,high):
    d=(pred-base)[high];r=(y-base)[high];return float(np.clip(d@r/max(d@d,1e-12),0,.25))
def stats(y,b,p,fold,seed=2026101673):
    draw=np.random.default_rng(seed).integers(0,len(y),(10000,len(y)));a=(b-y)**2;c=(p-y)**2;gain=s.rmse(b,y)-s.rmse(p,y);gg=np.sqrt(a[draw].mean(1))-np.sqrt(c[draw].mean(1));order=np.argsort(-(a-c))
    D=np.c_[np.ones(len(b)),b-b.mean()];coef=np.linalg.lstsq(D,p-b,rcond=None)[0];orth=p-b-D@coef
    return dict(n=len(y),baseline_rmse=s.rmse(b,y),candidate_rmse=s.rmse(p,y),gain=gain,conditional_CI95=np.quantile(gg,[.025,.975]).tolist(),two_primary_comparisons_CI975=np.quantile(gg,[.0125,.9875]).tolist(),
        fold_gains=[s.rmse(b[fold==k],y[fold==k])-s.rmse(p[fold==k],y[fold==k]) for k in range(5)],
        improved_folds=int(sum(s.rmse(p[fold==k],y[fold==k])<s.rmse(b[fold==k],y[fold==k])-1e-12 for k in range(5))),
        removed_largest_gains={str(n):dict(gain=s.rmse(b[order[n:]],y[order[n:]])-s.rmse(p[order[n:]],y[order[n:]]),share_net_SSE_gain=float((a-c)[order[:n]].sum()/(a-c).sum()) if abs((a-c).sum())>1e-12 else None) for n in [5,10]},
        n_helped=int((c<a-1e-10).sum()),n_harmed=int((c>a+1e-10).sum()),change_rms=float(np.sqrt(np.mean((p-b)**2))),
        affine_diagnostic=dict(intercept=float(coef[0]),slope_delta=float(coef[1]),affine_only_gain=s.rmse(b,y)-s.rmse(b+D@coef,y),orthogonal_only_gain=s.rmse(b,y)-s.rmse(b+orth,y),posthoc_diagnostic_not_fitted_candidate=True))
def predictions(k,stage,arm):
    frozen=arm.startswith('frozen');basearm={'frozen_ssl':'ssl','frozen_random':'none'}.get(arm,arm)
    return dict(np.load(_repo_io(s.HERE/f'models/fold{k}-{stage}-{basearm}'/('frozen-probe.npz' if frozen else 'predictions.npz'))))
def main():
    assert json.loads((s.HERE/'training-complete.json').read_text())['complete'];d=dict(np.load(_repo_io(s.OLD/'inputs.npz')));graphs=json.loads((s.OLD/'graphs.json').read_text());dev=d['dev'];y=d['y'][dev];b=d['base082'];fold=d['fold'];hi=d['noise'][dev]>=12
    p={a:b.copy() for a in ARMS};fixed={a:b.copy() for a in ARMS};direct={a:np.zeros(400) for a in ARMS};selection=[]
    for k,g in enumerate(graphs):
        z=np.load(_repo_io(s.OLD/f'anchors/fold{k}.npz'));A,B,T,V=[z[n] for n in ['A','B','T','V']];ix=z['ix'];rows={}
        for arm in ARMS:
            st=predictions(k,'selector',arm);rt=predictions(k,'refit',arm);assert np.array_equal(st['train'],A) and np.array_equal(st['query'],np.r_[B,V]);assert np.array_equal(rt['train'],T) and np.array_equal(rt['query'],V)
            aB=st['prediction'][:len(B)];aAV=st['prediction'][len(B):];aV=rt['prediction'];w=weight(aB,z['p82B'],d['y'][B],z['highB'])
            p[arm][ix]=b[ix]+z['highV']*w*(aV-b[ix]);fixed[arm][ix]=b[ix]+z['highV']*w*(aAV-b[ix]);direct[arm][ix]=aV
            dd=(aB-z['p82B'])[z['highB']];rr=(d['y'][B]-z['p82B'])[z['highB']];jj=np.random.default_rng(s.SEED+k).integers(0,len(dd),(2000,len(dd)));bw=np.clip((dd[jj]*rr[jj]).sum(1)/np.maximum((dd[jj]**2).sum(1),1e-12),0,.25)
            rows[arm]=dict(weight=w,coefficient_CI95_conditional_on_selected082=np.quantile(bw,[.025,.975]).tolist(),calibration_high_n=int(z['highB'].sum()),image_B_RMSE=s.rmse(aB,d['y'][B]),image_V_from_A_RMSE=s.rmse(aAV,d['y'][V]),image_V_from_T_RMSE=s.rmse(aV,d['y'][V]),refit_shift_rms=float(np.sqrt(np.mean((aV-aAV)**2))))
        selection.append(dict(fold=k,arms=rows))
    results={}
    for arm in ARMS:
        assert np.array_equal(p[arm][~hi],b[~hi]);r=stats(y,b,p[arm],fold);r['high_noise']=stats(y[hi],b[hi],p[arm][hi],fold[hi]);r['image_only_rmse']=s.rmse(direct[arm],y);r['high_noise_image_only_rmse']=s.rmse(direct[arm][hi],y[hi]);r['high_noise_error_correlation']=float(np.corrcoef((direct[arm]-y)[hi],(b-y)[hi])[0,1]);r['fixed_A_student']=stats(y,b,fixed[arm],fold);results[arm]=r
    matched={a:stats(y,p[a],p['ssl'],fold) for a in ['none','augment','long','frozen_ssl']};r=results['ssl'];checks=dict(overall_at_least_point03=r['gain']>=.03,high_noise_improves=r['high_noise']['gain']>0,three_folds=r['improved_folds']>=3,CI_vs082_positive=r['conditional_CI95'][0]>0,CI_vs_augmentation_positive=matched['augment']['conditional_CI95'][0]>0,top10_removed_positive=r['removed_largest_gains']['10']['gain']>0,beats_equal_updates=matched['long']['gain']>0)
    s.save(_repo_io('results.json'),dict(results=results,ssl_comparisons=matched,selection=selection,baseline076_rmse=s.rmse(d['base076'],y),baseline082_rmse=s.rmse(b,y),baseline082_high_rmse=s.rmse(b[hi],y[hi]),limitations=['Previously inspected400 labels; conditional bootstrap excludes historical model/search uncertainty and encoder refitting.','Inherited082 auxiliary representations were not newly trained within folds; new SSL, safety, downstream heads and blending are nested.','Only one initialization and fixed short pretraining budget. No private-leaderboard inference.']))
    s.save(_repo_io('decision.json'),dict(advance=all(checks.values()),checks=checks,incumbent='082',rollback='076',official_promotion=False))
    s.arrays('oof.npz',ids=d['ids'][dev],y=y,fold=fold,noise=d['noise'][dev],baseline082=b,baseline076=d['base076'],**p,**{a+'_image':v for a,v in direct.items()},**{a+'_fixedA':v for a,v in fixed.items()})
    with (s.HERE/'comparison.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','overall_RMSE','high_noise_RMSE','gain_vs082','CI95_low','CI95_high','improved_folds','gain_without_top10'])
        for a,r in results.items():w.writerow([a,r['candidate_rmse'],r['high_noise']['candidate_rmse'],r['gain'],*r['conditional_CI95'],r['improved_folds'],r['removed_largest_gains']['10']['gain']])
    with (s.HERE/'per-image.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','fold','noise','hardness','082',*ARMS,*[a+'_SSE_gain' for a in ARMS]])
        for i in range(400):w.writerow([d['ids'][dev[i]],fold[i],d['noise'][dev[i]],y[i],b[i],*[p[a][i] for a in ARMS],*[(b[i]-y[i])**2-(p[a][i]-y[i])**2 for a in ARMS]])
    print((s.HERE/'comparison.csv').read_text());print('ADVANCE',all(checks.values()),checks,flush=True)
if __name__=='__main__':
    sys.addaudithook(s.guard)
    with s.threadpool_limits(limits=2):main()
