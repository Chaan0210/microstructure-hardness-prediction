"""Frozen source-nested correction and complete-matrix evaluation."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import numpy as np,json,csv,sys
ARMS=['spatial','permuted','random']
def rmse(p,y):return float(np.sqrt(np.mean((p-y)**2)))
def coefficient(delta,y,base,high):
    d=delta[high];r=(y-base)[high];return float(np.clip(d@r/(d@d+len(d)),0,.5))
def metrics(y,b,p,fold,seed=2026101781):
    a=(b-y)**2;c=(p-y)**2;draw=np.random.default_rng(seed).integers(0,len(y),(10000,len(y)));gg=np.sqrt(a[draw].mean(1))-np.sqrt(c[draw].mean(1));order=np.argsort(-(a-c));D=np.c_[np.ones(len(y)),b-b.mean()];cf=np.linalg.lstsq(D,p-b,rcond=None)[0];orth=p-b-D@cf
    return dict(n=len(y),reference_rmse=rmse(b,y),candidate_rmse=rmse(p,y),gain=rmse(b,y)-rmse(p,y),conditional_CI95=np.quantile(gg,[.025,.975]).tolist(),conditional_CI9833=np.quantile(gg,[.008333,.991667]).tolist(),fold_gains=[rmse(b[fold==k],y[fold==k])-rmse(p[fold==k],y[fold==k]) for k in range(5)],improved_folds=int(sum(rmse(p[fold==k],y[fold==k])<rmse(b[fold==k],y[fold==k])-1e-12 for k in range(5))),
        removed_largest_gains={str(n):dict(gain=rmse(b[order[n:]],y[order[n:]])-rmse(p[order[n:]],y[order[n:]]),share_net_SSE_gain=float((a-c)[order[:n]].sum()/(a-c).sum()) if abs((a-c).sum())>1e-12 else None) for n in [5,10]},
        affine_diagnostic=dict(intercept=float(cf[0]),slope_delta=float(cf[1]),affine_only_gain=rmse(b,y)-rmse(b+D@cf,y),orthogonal_only_gain=rmse(b,y)-rmse(b+orth,y),posthoc_not_a_new_candidate=True),n_helped=int((c<a-1e-10).sum()),n_harmed=int((c>a+1e-10).sum()))
def main():
    assert json.loads((e.HERE/'readouts-complete.json').read_text())['complete'];d=dict(np.load(_repo_io(e.OLD/'inputs.npz')));g=json.loads((e.OLD/'graphs.json').read_text());dev=d['dev'];y=d['y'][dev];base=d['base082'];fold=d['fold'];high=(d['noise'][dev]>=12)&~d['failed'];p={a:base.copy() for a in ARMS};fixed={a:base.copy() for a in ARMS};seeds={f'{a}_seed{r}':base.copy() for a in ARMS for r in range(2)};image={a:np.zeros(400) for a in ARMS};phys=np.zeros(400);dirs={a:np.zeros(400) for a in ARMS};selection=[];geometry={a:[] for a in ['spatial','permuted']}
    for k,graph in enumerate(g):
        an=np.load(_repo_io(e.OLD/f'anchors/fold{k}.npz'));A,B,T,V=[an[n] for n in ['A','B','T','V']];ix=an['ix'];s=np.load(_repo_io(e.HERE/f'readouts/fold{k}-selector/predictions.npz'));v=np.load(_repo_io(e.HERE/f'readouts/fold{k}-refit/predictions.npz'));assert np.array_equal(s['train'],A) and np.array_equal(s['query'],np.r_[B,V]) and np.array_equal(v['train'],T) and np.array_equal(v['query'],V);row=dict(fold=k,n_calibration_high=int(an['highB'].sum()),arms={});phys[ix]=v['base']
        for arm in ARMS:
            ds=s[arm]-s['base'];dv=v[arm]-v['base'];ab=ds.mean(0)[:len(B)];av=ds.mean(0)[len(B):];vv=dv.mean(0);w=coefficient(ab,d['y'][B],an['p82B'],an['highB']);p[arm][ix]=base[ix]+an['highV']*w*vv;fixed[arm][ix]=base[ix]+an['highV']*w*av;image[arm][ix]=v[arm].mean(0);dirs[arm][ix]=vv
            individual=[]
            for r in range(2):
                wr=coefficient(ds[r,:len(B)],d['y'][B],an['p82B'],an['highB']);seeds[f'{arm}_seed{r}'][ix]=base[ix]+an['highV']*wr*dv[r];individual.append(wr)
            z=ab[an['highB']];rr=(d['y'][B]-an['p82B'])[an['highB']];draw=np.random.default_rng(2026101782+k).integers(0,len(z),(3000,len(z)));bw=np.clip((z[draw]*rr[draw]).sum(1)/((z[draw]**2).sum(1)+len(z)),0,.5)
            row['arms'][arm]=dict(beta=w,seed_betas=individual,beta_CI95_conditional_on082=np.quantile(bw,[.025,.975]).tolist(),calibration_direction_rms=float(np.sqrt(np.mean(z**2))),outer_direction_rms=float(np.sqrt(np.mean(vv[an['highV']]**2))),same_V_A_to_T_direction_shift_rms=float(np.sqrt(np.mean((vv-av)**2))))
        selection.append(row)
        for arm in geometry:
            arr=[];ref=[]
            for r in range(2):
                f=np.load(_repo_io(e.HERE/f'encoders/fold{k}-refit-seed{r}-{arm}/features.npz'));assert np.array_equal(f['query'],V);arr.append(f['geometry']);ref.append(f['geometry_reference'])
            geometry[arm].append(dict(fold=k,all_mean_DC_AC_gradient=np.mean(arr,axis=(0,1)).tolist(),high_mean_DC_AC_gradient=np.mean(np.array(arr)[:,an['highV']],axis=(0,1)).tolist(),reference_high_mean=np.mean(np.array(ref)[:,an['highV']],axis=(0,1)).tolist()))
    results={}
    for arm in ARMS:
        assert np.array_equal(p[arm][~high],base[~high]);r=metrics(y,base,p[arm],fold);r['high_noise']=metrics(y[high],base[high],p[arm][high],fold[high]);r['fixed_A_delta']=metrics(y,base,fixed[arm],fold);r['individual_encoder_seeds']=[metrics(y,base,seeds[f'{arm}_seed{j}'],fold) for j in range(2)];r['paired_tabular_components']=dict(physical_only_overall=rmse(phys,y),augmented_overall=rmse(image[arm],y),physical_only_high=rmse(phys[high],y[high]),augmented_high=rmse(image[arm][high],y[high]));results[arm]=r
    controls={a:metrics(y,p[a],p['spatial'],fold) for a in ['permuted','random']};r=results['spatial'];checks=dict(overall_gain_at_least05=r['gain']>=.05,high_noise_gain_positive=r['high_noise']['gain']>0,three_folds=r['improved_folds']>=3,CI_vs082_positive=r['conditional_CI95'][0]>0,CI_vs_permuted_positive=controls['permuted']['conditional_CI95'][0]>0,CI_vs_random_positive=controls['random']['conditional_CI95'][0]>0,top10_removed_positive=r['removed_largest_gains']['10']['gain']>0,both_encoder_seeds_nonnegative=all(x['gain']>=0 for x in r['individual_encoder_seeds']))
    e.save(_repo_io('results.json'),dict(results=results,spatial_vs_controls=controls,selection=selection,geometry=geometry,baseline082_overall=rmse(base,y),baseline082_high=rmse(base[high],y[high]),limitations=['Previously exposed development400; source bootstrap conditional on fixed fits.','New encoders/scalers/PCA/Tab readouts and corrections are nested; inherited082 auxiliary features/historical selection are not newly nested.','Per-source masked image reconstruction is not ground-truth grain segmentation or proof of unbiased denoising.']))
    e.save(_repo_io('decision.json'),dict(advance=all(checks.values()),checks=checks,production_beta_if_advanced=float(np.mean([z['arms']['spatial']['beta'] for z in selection])),incumbent='082',rollback='076',official_promotion=False))
    e.arrays('oof.npz',ids=d['ids'][dev],y=y,fold=fold,high=high,noise=d['noise'][dev],baseline082=base,physical=phys,**p,**seeds,**{a+'_augmented':v for a,v in image.items()},**{a+'_delta':v for a,v in dirs.items()})
    with (e.HERE/'comparison.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','overall_RMSE','high_RMSE','gain_vs082','CI95_low','CI95_high','improved_folds','top10_removed_gain','seed0_gain','seed1_gain'])
        for a,r in results.items():w.writerow([a,r['candidate_rmse'],r['high_noise']['candidate_rmse'],r['gain'],*r['conditional_CI95'],r['improved_folds'],r['removed_largest_gains']['10']['gain'],*[z['gain'] for z in r['individual_encoder_seeds']]])
    with (e.HERE/'per-image.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','fold','noise','hardness','082',*ARMS,*[a+'_SSE_gain' for a in ARMS]])
        for i in range(400):w.writerow([d['ids'][dev[i]],fold[i],d['noise'][dev[i]],y[i],base[i],*[p[a][i] for a in ARMS],*[(base[i]-y[i])**2-(p[a][i]-y[i])**2 for a in ARMS]])
    print((e.HERE/'comparison.csv').read_text());print('ADVANCE',all(checks.values()),checks,flush=True)
if __name__=='__main__':
    sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
