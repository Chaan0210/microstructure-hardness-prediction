"""Frozen complete-matrix nested evaluation; no outer-selected control winner."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import stats as s
import numpy as np,json,csv,sys
ARMS=['topology','local','hardness','random','raw']
def main():
    assert json.loads((c.HERE/'readouts-complete.json').read_text())['complete'];assert json.loads((c.HERE/'pre-score-verification.json').read_text())['all_passed'];d=c.inputs(labels=True);dev=d['dev'];y=d['y'][dev];base=d['base082'];fold=d['fold'];high=(d['noise'][dev]>=12)&~d['failed'];p={a:base.copy() for a in ARMS};fixed={a:base.copy() for a in ARMS};seed={f'{a}_seed{r}':base.copy() for a in ARMS for r in range(2)};directions={a:np.zeros(400) for a in ARMS};aug={a:np.zeros(400) for a in ARMS};physical=np.zeros(400);selection=[];weak={a:np.zeros((400,3)) for a in ARMS[:3]}
    for k,g in enumerate(d['graphs']):
        an=np.load(_repo_io(c.OLD/f'anchors/fold{k}.npz'));B=an['B'];V=an['V'];ix=an['ix'];a=np.load(_repo_io(c.HERE/f'readouts/fold{k}-selector/predictions.npz'));v=np.load(_repo_io(c.HERE/f'readouts/fold{k}-refit/predictions.npz'));assert np.array_equal(a['train'],an['A']) and np.array_equal(a['query'],np.r_[B,V]) and np.array_equal(v['train'],an['T']) and np.array_equal(v['query'],V);physical[ix]=v['base'];row=dict(fold=k,n_high_B=int(an['highB'].sum()),arms={})
        for arm in ARMS:
            db=a[arm]-a['base'];dv=v[arm]-v['base'];dbmean=db.mean(0);dm=dv.mean(0);coef=s.coefficient(dbmean[:len(B)],d['y'][B],an['p82B'],an['highB']);p[arm][ix]=base[ix]+an['highV']*coef*dm;fixed[arm][ix]=base[ix]+an['highV']*coef*dbmean[len(B):];directions[arm][ix]=dm;aug[arm][ix]=v[arm].mean(0);ss=[]
            for r in range(2):
                beta=s.coefficient(db[r,:len(B)],d['y'][B],an['p82B'],an['highB']);seed[f'{arm}_seed{r}'][ix]=base[ix]+an['highV']*beta*dv[r];ss.append(beta)
            z=dbmean[:len(B)][an['highB']];res=(d['y'][B]-an['p82B'])[an['highB']];draw=np.random.default_rng(2026101794+k).integers(0,len(z),(3000,len(z)));ww=np.clip((z[draw]*res[draw]).sum(1)/((z[draw]**2).sum(1)+len(z)),0,.5)
            row['arms'][arm]=dict(beta=coef,seed_betas=ss,conditional_beta_CI95=np.quantile(ww,[.025,.975]).tolist(),B_direction_rms=float(np.sqrt(np.mean(z**2))),V_direction_rms=float(np.sqrt(np.mean(dm[an['highV']]**2))),same_V_A_to_T_shift_rms=float(np.sqrt(np.mean((dm-dbmean[len(B):])**2))))
            if arm in weak:
                ff=[np.load(_repo_io(c.HERE/f'encoders/fold{k}-refit-seed{r}-{arm}/features.npz'))['weak_query_losses'] for r in range(2)];weak[arm][ix]=np.mean(ff,0)
        selection.append(row)
    results={}
    for arm in ARMS:
        assert np.array_equal(p[arm][~high],base[~high]);r=s.metrics(y,base,p[arm],fold,seed=2026101795);r['high_noise']=s.metrics(y[high],base[high],p[arm][high],fold[high],seed=2026101795);r['seeds']=[s.metrics(y,base,seed[f'{arm}_seed{i}'],fold,seed=2026101795) for i in range(2)];r['fixed_A_branch']=s.metrics(y,base,fixed[arm],fold,seed=2026101795);r['paired_tabular_components']=dict(physical_overall=s.rmse(physical,y),augmented_overall=s.rmse(aug[arm],y),physical_high=s.rmse(physical[high],y[high]),augmented_high=s.rmse(aug[arm][high],y[high]));results[arm]=r
    comparisons={a:s.metrics(y,p[a],p['topology'],fold,seed=2026101795) for a in ARMS[1:]};r=results['topology'];checks=dict(overall_gain05=r['gain']>=.05,high_gain=r['high_noise']['gain']>0,three_folds=r['improved_folds']>=3,CI_vs082=r['conditional_CI95'][0]>0,CI_vs_each_control=all(z['conditional_CI95'][0]>0 for z in comparisons.values()),top10=r['removed_largest_gains']['10']['gain']>0,both_seeds=all(z['gain']>=0 for z in r['seeds']))
    structural={a:dict(overall_BCE_boundary_local_connection=v.mean(0).tolist(),high_BCE_boundary_local_connection=v[high].mean(0).tolist(),balanced_constant_reference=float(np.log(2))) for a,v in weak.items()}
    c.save(_repo_io('results.json'),dict(baseline082_overall=s.rmse(base,y),baseline082_high=s.rmse(base[high],y[high]),results=results,topology_vs_controls=comparisons,selection=selection,weak_structure=structural,limitations=['Previously exposed400 development sources; no untouched confirmation.','New fits nested conditional on historical incumbent representations.','Pseudo-boundary/connectivity targets are auxiliary predictions, not true grain labels.','Source-bootstrap intervals are conditional on fixed fitted predictions; training sets overlap.','Raw control has one three-Tab-seed readout, duplicated only for a common array schema, not two independent encoder seeds.']))
    c.save(_repo_io('decision.json'),dict(advance=all(checks.values()),checks=checks,incumbent='082',rollback='076',official_promotion=False,production_beta_if_advanced=float(np.mean([z['arms']['topology']['beta'] for z in selection]))));c.arrays('oof.npz',ids=d['ids'][dev],y=y,fold=fold,high=high,noise=d['noise'][dev],baseline082=base,physical=physical,**p,**seed,**{a+'_delta':v for a,v in directions.items()},**{a+'_augmented':v for a,v in aug.items()},**{a+'_weak':v for a,v in weak.items()})
    with (c.HERE/'comparison.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','overall_RMSE','high_RMSE','gain_vs082','CI95_low','CI95_high','improved_folds','top10_removed_gain','seed0_gain','seed1_gain'])
        for arm,r in results.items():w.writerow([arm,r['candidate_rmse'],r['high_noise']['candidate_rmse'],r['gain'],*r['conditional_CI95'],r['improved_folds'],r['removed_largest_gains']['10']['gain'],*[z['gain'] for z in r['seeds']]])
    with (c.HERE/'per-image.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','fold','noise','hardness','082',*ARMS,*[a+'_SSE_gain' for a in ARMS]])
        for j in range(400):w.writerow([d['ids'][dev[j]],fold[j],d['noise'][dev[j]],y[j],base[j],*[p[a][j] for a in ARMS],*[(base[j]-y[j])**2-(p[a][j]-y[j])**2 for a in ARMS]])
    print((c.HERE/'comparison.csv').read_text());print('ADVANCE',all(checks.values()),checks,flush=True)
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
